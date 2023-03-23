import logging
import ray
import copy
import chess
import datetime
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from config.config import Config, interpolate
from src_code.agent.agent import AlphaZeroChess, board_to_input, create_network
from src_code.agent.utils import draw_board, visualize_tree, get_board_piece_count, generate_game_id, save_training_data, load_training_data, scan_redis_for_training_data

USE_RAY = False
if USE_RAY:
    NUM_WORKERS = mp.cpu_count() - 2
    ray.init(num_cpus=NUM_WORKERS, num_gpus=0, ignore_reinit_error=True, logging_level=logging.DEBUG)

logging.getLogger('tensorflow').setLevel(logging.WARNING)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    gpu_idx = 0  # Set the index of the GPU you want to use
    tf.config.experimental.set_virtual_device_configuration(physical_devices[gpu_idx], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 1)])  # Set the memory limit (in bytes)
else:
    print('No GPUs available')


def play_games(pass_dict):
    game_id = pass_dict['game_id']
    key_prefix = pass_dict['key_prefix']
    num_iterations = pass_dict['num_iterations']
    # Initialize the config and agent
    config = Config(num_iterations, verbosity=False)

    # Play the game
    for i in range(config.self_play_games):
        agent = AlphaZeroChess(config)

        key_id_list = []
        states = []
        policy_targets = []

        # training loop:
        while not agent.game_over():
            # Get the current state of the board
            player = 'white' if agent.board.turn else 'black'
            uci_move, policy, policy_target = agent.get_action()

            # Take the action and update the board state
            agent.board.push_uci(uci_move)

            key_id = f'{key_prefix}_{game_id}_{agent.move_counter.count}'
            key_id_list.append(key_id)
            # Print the board
            print(f'The {agent.move_counter.count} move was: {uci_move}')
            if agent.move_counter.count > agent.temperature_threshold:
                agent.update_temperature()
            if (agent.move_counter.count % 1) == 0 and (agent.move_counter.count > 0):
                agent.tree.width()
                print(f'Piece count (white / black): {get_board_piece_count(agent.board)}')
                print(agent.board)
                # if (agent.move_counter.count % 50) == 0:
                #    draw_board(agent.board, display=True, verbosity=True)
            # Append the training data
            state = board_to_input(config, agent.board)
            states.append(state)
            policy_targets.append(policy_target)

            # Update the tree
            agent.tree.update_root(uci_move)
            agent.move_counter.increment()

            # Print the result of the game
            if agent.game_over() or agent.move_counter.count > config.maximum_moves:
                print(f'Game Over! Winner is {agent.board.result()}')
                # add the value outcomes to the training data
                value_target = None

                if agent.board.result() == '1-0':
                    value_target = 1
                elif agent.board.result() == '0-1':
                    value_target = -1
                elif agent.board.result() == '1/2-1/2':
                    # modify for white players
                    if player == 'white':
                        value_target = 0.25
                    else:
                        value_target = -0.25
                else:
                    value_target = 0

                value_targets = [value_target * config.reward_discount ** (len(policy_targets) - i) for i in range(len(policy_targets) + 1)]

                # Update the game counter
                config.game_counter.increment()

                for j, key_id in enumerate(key_id_list):
                    key_dict = dict()
                    key_dict['key'] = key_id
                    key_dict['game_id'] = game_id
                    key_dict['move_id'] = j
                    key_dict['state'] = states[j]
                    key_dict['policy_target'] = policy_targets[j]
                    key_dict['value_target'] = value_targets[j]

                    # Save the training data
                    save_training_data(agent, key_id, key_dict)




def train_model(key_prefix):
    config = Config(num_iterations=None, verbosity=False)
    agent = AlphaZeroChess(config)

    key_list = scan_redis_for_training_data(agent, key_prefix)
    cwc = 0


@ray.remote
def main_ray(in_params):
    return main(in_params)


def main(in_params):
    print(f'in_params: {in_params}')
    type = in_params['type']
    num_iterations = in_params['num_iterations']
    print(f'Running the main function with type: {type}')
    key_prefix = 'azChess_ThreadripperData_test'

    if type == 'create_training_data':
        game_id = generate_game_id()
        pass_dict = dict()
        pass_dict['game_id'] = game_id
        pass_dict['key_prefix'] = key_prefix
        pass_dict['num_iterations'] = num_iterations
        play_games(pass_dict)

    elif type == 'train':
        train_model(key_prefix)

    return f'Finished running the main function with type: {type} Game ID: {game_id}'


def initialize(in_config):
    network = create_network(in_config)
    outer_agent = AlphaZeroChess(in_config, network=network)
    outer_agent.save_networks('network_current')


if __name__ == '__main__':
    type_list = ['initialize', 'create_training_data', 'train']
    type_id = 1

    min_iterations = 2000
    outer_config = Config(num_iterations=min_iterations, verbosity=False)

    if type_list[type_id] == 'initialize':
        initialize(outer_config)

    if type_list[type_id] != 'initialize' and USE_RAY is False:
        params = dict()
        params['type'] = type_list[type_id]
        params['num_iterations'] = min_iterations
        outcome = main(params)
        print(f'Outcome: {outcome}')
    elif type_list[type_id] != 'initialize' and USE_RAY is True:

        max_num_iterations = 1600

        outer_config = Config(min_iterations, verbosity=False)

        start_ind = 0
        while start_ind < outer_config.self_play_games:
            inds = list(range(start_ind, min(start_ind + NUM_WORKERS, outer_config.self_play_games)))
            params_list = []

            for ind in inds:
                # Get the type ID for the current index
                params_item = dict()
                params_item['type'] = type_list[type_id]
                params_item['num_iterations'] = int(0.5 + interpolate(min_iterations, max_num_iterations, (min(ind, 5000)/5000)))
                params_list.append(params_item)

            results = [main_ray.remote(params_list[j]) for j in range(len(inds))]

            # Wait for all tasks to complete and get the results
            output = ray.get(results)

            # Print the output of each task
            for i, result in enumerate(output):
                print(f'Task {i} output: {result}')

            start_ind += NUM_WORKERS

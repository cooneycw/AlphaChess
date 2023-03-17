import logging
import ray
import copy
import chess
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from config.config import Config
from src_code.agent.agent import AlphaZeroChess, board_to_input
from src_code.agent.utils import draw_board, visualize_tree, get_board_piece_count

NUM_WORKERS = mp.cpu_count() - 2

# ray.init(num_cpus=NUM_WORKERS, num_gpus=0, ignore_reinit_error=True, logging_level=logging.DEBUG)

logging.getLogger('tensorflow').setLevel(logging.WARNING)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    gpu_idx = 0  # Set the index of the GPU you want to use
    tf.config.experimental.set_virtual_device_configuration(physical_devices[gpu_idx], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 1)])  # Set the memory limit (in bytes)
else:
    print('No GPUs available')


def play_games(game_id):
    # Initialize the config and agent
    config = Config(verbosity=False)
    agent = AlphaZeroChess(config)

    # Initialize the training data
    key_id_list = []
    states_white = []
    states_black = []
    policy_targets_white = []
    policy_targets_black = []
    value_targets_white = []
    value_targets_black = []
    # Play the game
    agent.game_counter.reset()
    for i in range(config.self_play_games):
        policy_targets_temp_white = []
        policy_targets_temp_black = []
        # training loop:
        agent.move_counter.reset()
        while agent.game_counter.count <= config.self_play_games:
            while not agent.game_over():
                # Get the current state of the board
                player = 'white' if agent.board.turn else 'black'
                uci_move, policy, policy_target = agent.get_action()

                # Take the action and update the board state
                agent.board.push_uci(uci_move)

                key_id = f'{game_id}_{agent.move_counter.count}'
                key_id_list.append(key_id)
                # Print the board
                print(f'The {agent.move_counter.count} move was: {uci_move}')
                if (agent.move_counter.count % 10) == 0 and (agent.move_counter.count > 0):
                    agent.tree.width()
                    print(f'Piece count (white / black): {get_board_piece_count(agent.board)}')
                    if (agent.move_counter.count % 50) == 0:
                        draw_board(agent.board, display=True, verbosity=True)
                if player == 'white':
                    # Append the training data
                    state = board_to_input(config, agent.board)
                    states_white.append(state)
                    policy_targets_temp_white.append(policy_target)
                else:
                    # Append the training data
                    state = board_to_input(config, agent.board)
                    states_black.append(state)
                    policy_targets_temp_black.append(policy_target)

                # Update the tree
                agent.tree.update_root(uci_move)
                agent.move_counter.increment()

                # Print the result of the game
                if agent.game_over():
                    print(f'Game Over! Winner is {agent.board.result()}')
                    # add the value outcomes to the training data
                    value_target_white = None
                    value_target_black = None

                    if agent.board.result() == '1-0':
                        if agent.tree.root.player_to_move == 'black':
                            value_target_white = 1
                            value_target_black = -1
                    elif agent.board.result() == '0-1':
                        if agent.tree.root.player_to_move == 'white':
                            value_target_white = -1
                            value_target_black = 1
                    elif agent.board.result() == '1/2-1/2':
                            value_target_white = 0.25
                            value_target_black = 0.25

                    value_targets_white = [value_target_white * config.reward_discount ** (len(policy_targets_temp_white) - i) for i in range(len(policy_targets_temp_white) + 1)]
                    value_targets_black = [value_target_black * config.reward_discount ** (len(policy_targets_temp_black) - i) for i in range(len(policy_targets_temp_black) + 1)]
                    policy_targets_white.append(policy_targets_temp_white)
                    policy_targets_black.append(policy_targets_temp_black)

            # Train the network
            agent.update_network_white(states_white, policy_targets_white, value_targets_white)
            agent.update_network_black(states_black, policy_targets_black, value_targets_black)

    # Save the final weights
    agent.save_network_weights(key_name='agent_network_weights')


#@ray.remote
def main():
    play_games()
    return f'Finished!'


if __name__ == '__main__':
    config = Config(verbosity=False)
    main()

    start_ind = 0
    while start_ind < config.self_play_games:
        inds = list(range(start_ind, min(start_ind + NUM_WORKERS, config.self_play_games)))

        results = [main.remote() for _ in range(len(inds))]

        # Wait for all tasks to complete and get the results
        output = ray.get(results)

        # Print the output of each task
        for i, result in enumerate(output):
            print(f'Task {i} output: {result}')

        start_ind += NUM_WORKERS


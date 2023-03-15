import logging
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from config.config import Config
from src_code.agent.agent import AlphaZeroChess, board_to_input
from src_code.agent.utils import draw_board, visualize_tree


logging.getLogger('tensorflow').setLevel(logging.WARNING)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    gpu_idx = 0  # Set the index of the GPU you want to use
    tf.config.experimental.set_virtual_device_configuration(physical_devices[gpu_idx], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])  # Set the memory limit (in bytes)
else:
    print('No GPUs available')


def main():
    config = Config(verbosity=False)
    play_games(config)


def play_games(config):
    # Initialize the config and agent

    agent = AlphaZeroChess(config)

    # Initialize the training data
    states = []
    policy_targets = []
    value_targets = []
    # Play the game
    agent.game_counter.reset()
    for i in range(config.self_play_games):
        # training loop:
        agent.move_counter.reset()
        while agent.game_counter.count <= config.self_play_games:
            while not agent.game_over():
                # Get the current state of the board

                uci_move, policy_target = agent.get_action()

                # Take the action and update the board state
                agent.board.push_uci(uci_move)

                # Print the board
                print(f'The {agent.move_counter.count} move was: {uci_move}')
                if (agent.sim_counter.count % 100) == 0 and (agent.sim_counter.count > 0):
                    draw_board(agent.board, display=True, verbosity=True)
                policy_target = agent.tree
                value_target = agent.get_result()

                # Append the training data
                state = board_to_input(config, agent.board)
                states.append(state)
                policy_targets.append(policy_target)
                value_targets.append(value_target)

                # Update the tree
                agent.tree.update_root(uci_move)
                agent.move_counter.increment()

                # Print the result of the game
                if agent.game_over():
                    print(f'Game Over! Winner is {agent.board.result()}')
                    agent.reset()

            agent.move_counter.reset()
            agent.game_counter.increment()

            # Train the network
            agent.update_network(states, policy_targets, value_targets)

    # Save the final weights
    agent.save_network_weights(key_name='agent_network_weights')


if __name__ == '__main__':
    main()
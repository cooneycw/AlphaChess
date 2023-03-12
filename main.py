import tensorflow as tf
import multiprocessing as mp
from config.config import Config
from src_code.agent.agent import AlphaZeroChess, board_to_input, generate_training_data
from src_code.agent.utils import draw_board, visualize_tree

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

    # Play the game
    for i in range(config.self_play_games):
        while not agent.game_over():
            # Get the current state of the board
            state = board_to_input(config, agent.board)

            # Get the best action to take
            action = agent.get_action(state)

            # Take the action and update the board state
            uci_move = config.all_chess_moves[action]
            agent.board.push_uci(uci_move)

            # Print the board
            print(f'Move was: {uci_move}')
            draw_board(agent.board, display=True, verbosity=True)

            # Generate training data
            states, policy_targets, value_targets = generate_training_data(agent, config)

            # Train the network
            agent.update_network(states, policy_targets, value_targets)
            agent.update_temperature()

    # Save the final weights
    agent.save_network_weights(key_name='agent_network_weights')


if __name__ == '__main__':
    main()
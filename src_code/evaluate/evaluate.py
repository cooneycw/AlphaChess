import chess
import numpy as np
import tensorflow as tf
from config.config import Config
from src_code.agent.agent import AlphaZeroChess, board_to_input
from src_code.evaluate.utils import scan_redis_for_networks, load_network


def evaluate_network(in_dict):
    num_evals = in_dict['num_evals']
    network_prefix = in_dict['network_prefix']
    config = Config(num_iterations=1600, verbosity=False)
    agent_current = AlphaZeroChess(config)
    agent_candidate = AlphaZeroChess(config)

    network_keys = scan_redis_for_networks(agent_current, network_prefix)

    num_wins = 0
    num_losses = 0
    num_draws = 0

    for i in range(num_games):
        board = chess.Board()
        is_agent_turn = np.random.choice([True, False])

        while not board.is_game_over():
            if is_agent_turn:
                state = board_to_input(board)
                policy_logits, value = agent.predict(np.expand_dims(state, axis=0))
                legal_moves = list(board.legal_moves)
                policy = tf.nn.softmax(policy_logits[0]).numpy()
                action = np.random.choice(len(legal_moves), p=policy[legal_moves])
                board.push(legal_moves[action])
            else:
                state = board_to_input(board)
                policy_logits, value = opponent.predict(np.expand_dims(state, axis=0))
                legal_moves = list(board.legal_moves)
                policy = tf.nn.softmax(policy_logits[0]).numpy()
                action = np.random.choice(len(legal_moves), p=policy[legal_moves])
                board.push(legal_moves[action])

            is_agent_turn = not is_agent_turn

        result = board.result()
        if result == '1-0':
            num_wins += 1
        elif result == '0-1':
            num_losses += 1
        else:
            num_draws += 1

        print(f"Game {i+1}/{num_games}: {result}")

    print(f"Results: {num_wins} wins, {num_losses} losses, {num_draws} draws")


def get_networks():
    config = Config(num_iterations=None, verbosity=False)
    agent = AlphaZeroChess(config)
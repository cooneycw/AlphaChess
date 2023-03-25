import chess
import random
import numpy as np
import tensorflow as tf
from config.config import Config
from src_code.agent.agent import AlphaZeroChess, board_to_input
from src_code.agent.utils import get_board_piece_count
from src_code.evaluate.utils import scan_redis_for_networks, delete_redis_key


def evaluate_network(in_dict):
    config = Config(num_iterations=1600, verbosity=False)
    num_evals = in_dict['num_evals']
    network_prefix = in_dict['network_prefix']
    agent_admin = AlphaZeroChess(config)
    network_keys = scan_redis_for_networks(agent_admin, network_prefix)
    key = network_keys[0]
    print(f'Evaluating network {key}')

    game_cnt = 0
    challenger_wins = 0
    challenger_losses = 0
    challenger_draws = 0

    for i in range(num_evals):
        if random.random() < 0.5:
            player_to_go = 'current'
        else:
            player_to_go = 'candidate'
        starting_player = player_to_go
        print(f'Game {i} Player to go: {player_to_go}')
        board = chess.Board()
        agent_current = AlphaZeroChess(config)
        agent_candidate = AlphaZeroChess(config)
        agent_candidate.load_networks(key)
        move_cnt = 0

        while not board.is_game_over():
            if player_to_go == 'current':
                uci_move, _, _ = agent_current.get_action()
                _, _, _ = agent_candidate.get_action(iters=120)
                player_to_go = 'candidate'
            else:
                uci_move, _, _ = agent_candidate.get_action()
                _, _, _ = agent_current.get_action(iters=120)
                player_to_go = 'current'

            board.push_uci(uci_move)
            move_cnt += 1
            if move_cnt > 20:
                agent_current.update_temperature()
                agent_candidate.update_temperature()
            agent_current.board.push_uci(uci_move)
            agent_candidate.board.push_uci(uci_move)
            agent_current.tree.update_root(uci_move)
            agent_candidate.tree.update_root(uci_move)
            result = board.result()
            print(f'The {move_cnt} move was: {uci_move}')
            print(f'Piece count (white / black): {get_board_piece_count(board)} White: {starting_player}')
            print(board)
            if board.is_game_over(claim_draw=True):
                if result == '1-0':
                    if player_to_go == 'current':
                        challenger_wins += 1
                    else:
                        challenger_losses += 1
                elif result == '0-1':
                    if player_to_go == 'current':
                        challenger_losses += 1
                    else:
                        challenger_wins += 1
                else:
                    challenger_draws += 1
                game_cnt += 1
                print(f'Games: {game_cnt} {challenger_wins / game_cnt} Challenger wins: {challenger_wins} Losses: {challenger_losses} Draws: {challenger_draws}')

    # Save candidate network if it wins and delete it if it loses
    if (challenger_wins / game_cnt) > 0.55:
        print(f'Challenger won {challenger_wins / game_cnt} of the games')
        agent_candidate.save_networks('current_network')
        delete_redis_key(agent_admin, key)


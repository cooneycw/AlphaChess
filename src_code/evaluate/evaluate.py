import random
import copy
import chess
import ray
import numpy as np
import tensorflow as tf
from config.config import Config
from src_code.agent.agent import AlphaZeroChess, board_to_input
from src_code.agent.utils import get_board_piece_count


@ray.remote
def run_evaluation(game_id, key):

    config = Config(num_iterations=1600, verbosity=False)
    if random.random() < 0.5:
        player_to_go = 'current'
    else:
        player_to_go = 'candidate'
    starting_player = player_to_go
    print(f'Game {game_id} Player to go: {player_to_go}')
    board = chess.Board()
    agent_current = AlphaZeroChess(config)
    agent_candidate = AlphaZeroChess(config)
    agent_candidate.load_networks(key)
    move_cnt = 0

    while not board.is_game_over(claim_draw=True):
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

        out_params = dict()

        out_params['challenger_wins'] = 0
        out_params['challenger_losses'] = 0
        out_params['challenger_draws'] = 0

        if board.is_game_over(claim_draw=True) or move_cnt > 150:
            if result == '1-0':
                if player_to_go == 'current':
                    out_params['challenger_wins'] = 1
                else:
                    out_params['challenger_losses'] = 1
            elif result == '0-1':
                if player_to_go == 'current':
                    out_params['challenger_losses'] = 1
                else:
                    out_params['challenger_wins'] = 1
            else:
                out_params['challenger_draws'] = 1

            return out_params



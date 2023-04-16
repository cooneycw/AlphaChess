import random
import copy
import gc
import chess
import ray
import numpy as np
import tensorflow as tf
from config.config import Config
from src_code.agent.agent import AlphaZeroChess, board_to_input
from src_code.agent.utils import get_board_piece_count, malloc_trim


@ray.remote
def run_evaluation(game_id, key):

    config = Config(num_iterations=1200, verbosity=False)
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

    out_params = dict()

    out_params['challenger_wins'] = 0
    out_params['challenger_losses'] = 0
    out_params['challenger_draws'] = 0

    move_cnt = 0

    while not board.is_game_over(claim_draw=True):
        if player_to_go == 'current':
            uci_move, _, _ = agent_current.get_action()
            _, _, _ = agent_candidate.get_action()
            player_to_go = 'candidate'
        else:
            uci_move, _, _ = agent_candidate.get_action()
            _, _, _ = agent_current.get_action()
            player_to_go = 'current'

        board.push_uci(uci_move)
        move_cnt += 1
        if move_cnt > 20:
            agent_current.update_temperature()
            agent_candidate.update_temperature()
        old_node_list_current = agent_current.tree.root.get_all_nodes()
        old_node_list_candidate = agent_candidate.tree.root.get_all_nodes()
        agent_current.board.push_uci(uci_move)
        agent_candidate.board.push_uci(uci_move)
        agent_current.tree.update_root(uci_move)
        agent_candidate.tree.update_root(uci_move)

        new_node_list_current = agent_current.tree.root.get_all_nodes()
        new_node_list_candidate = agent_candidate.tree.root.get_all_nodes()

        # Update the tree
        for abandoned_node_current in set(old_node_list_current).difference(set(new_node_list_current)):
            abandoned_node_current.remove_from_all_nodes()
            del abandoned_node_current

        for abandoned_node_candidate in set(old_node_list_candidate).difference(set(new_node_list_candidate)):
            abandoned_node_candidate.remove_from_all_nodes()
            del abandoned_node_candidate

        del old_node_list_current, new_node_list_current, old_node_list_candidate, new_node_list_candidate

        gc.collect()
        malloc_trim()

        result = board.result()
        print(f'The {move_cnt} move was: {uci_move}')
        print(f'Piece count (white / black): {get_board_piece_count(board)} White: {starting_player}')
        print(board)

        if board.is_game_over(claim_draw=True) or move_cnt > config.maximum_moves:
            if result == '1-0':
                if starting_player == 'candidate':
                    print('Challenger wins!')
                    out_params['challenger_wins'] += 1
                else:
                    print('Current wins!')
                    out_params['challenger_losses'] += 1
            elif result == '0-1':
                if starting_player == 'candidate':
                    print('Current wins!')
                    out_params['challenger_losses'] += 1
                else:
                    print('Challenger wins!')
                    out_params['challenger_wins'] += 1
            else:
                print('Draw..')
                out_params['challenger_draws'] += 1

            agent_current.tree = None
            agent_candidate.tree = None
            agent_current = None
            agent_candidate = None
            del agent_current, agent_candidate
            gc.collect()
            malloc_trim()

            return out_params

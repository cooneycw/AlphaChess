import random
import copy
import gc
import datetime
import chess
import ray
import numpy as np
import tensorflow as tf
from config.config import Config
from src_code.agent.agent import AlphaZeroChess, board_to_input, Node
from src_code.agent.utils import get_board_piece_count, malloc_trim


def run_evaluation(in_params):
    verbosity = in_params['verbosity']
    rand_val = in_params['rand_val']
    game_id = in_params['eval_game_id']
    network_current = in_params['network_current']
    network_candidate = in_params['network_candidate']

    config = Config(verbosity=verbosity)
    if rand_val < 0.5:
        player_to_go = 'current'
    else:
        player_to_go = 'candidate'
    starting_player = player_to_go
    print(f'Game {game_id} Player to go: {player_to_go}')
    board = chess.Board()
    agent_current = AlphaZeroChess(config)
    agent_current.load_networks(network_current)
    agent_candidate = AlphaZeroChess(config)
    agent_candidate.load_networks(network_candidate)

    out_params = dict()

    out_params['player_to_go'] = starting_player
    out_params['challenger_wins'] = 0
    out_params['challenger_losses'] = 0
    out_params['challenger_draws'] = 0

    move_cnt = 0

    while not board.is_game_over(claim_draw=True):
        if player_to_go == 'current':
            uci_move, _, _ = agent_current.get_action(eval=True)
            _, _, _ = agent_candidate.get_action(eval=True)
            player_to_go = 'candidate'
        else:
            uci_move, _, _ = agent_candidate.get_action(eval=True)
            _, _, _ = agent_current.get_action(eval=True)
            player_to_go = 'current'

        # print(f'\n \nCurrent node statistics:')
        # agent_current.tree.gather_tree_statistics()
        # print(f'\n \nCandidate node statistics')
        # agent_candidate.tree.gather_tree_statistics()

        board.push_uci(uci_move)
        move_cnt += 1
        if move_cnt > agent_current.temperature_threshold:
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

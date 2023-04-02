import gc
import inspect
import sys
import tracemalloc
from chess import Board
from config.config import Config
from src_code.agent.agent import AlphaZeroChess, Node
from src_code.agent.chess_env import ChessGame
from src_code.agent.agent import board_to_input, draw_board
from src_code.agent.utils import get_board_piece_count, save_training_data, get_var_sizes, \
    print_variable_sizes_pympler, print_uncollected_objects


def play_games(pass_dict):
    game_id = pass_dict['game_id']
    key_prefix = pass_dict['key_prefix']
    num_iterations = pass_dict['num_iterations']
    self_play_games = pass_dict['self_play_games']
    # Initialize the config and agent
    config = Config(num_iterations, verbosity=False)

    # Play the game
    for i in range(self_play_games):
        agent = AlphaZeroChess(config)
        chess_game_play = ChessGame()

        key_id_list = []
        states = []
        policy_targets = []
        game_limit_stop = False
        # training loop:
        while not agent.game_over() and not game_limit_stop:
            # Get the current state of the board
            player = 'white' if agent.chess_game_agent.board.turn else 'black'
            uci_move, policy, policy_target = agent.get_action()

            # agent.tree.root.count_nodes()
            # print(f'Global variables: {print_variable_sizes_pympler(globals())}')
            # print(f'Local variables: {print_variable_sizes_pympler(locals())}')

            # Take the action and update the board state
            # print(uci_move)
            # if player == 'black':
            #     uci_move = input()

            agent.chess_game_agent.board.push_uci(uci_move)

            key_id = f'{key_prefix}_{game_id}_{agent.move_counter.count}'
            key_id_list.append(key_id)
            # Print the board
            print(f'The {agent.move_counter.count} move was: {uci_move}')
            if agent.move_counter.count > agent.temperature_threshold:
                agent.update_temperature()
            if (agent.move_counter.count % 1) == 0 and (agent.move_counter.count > 0):
                agent.tree.width()
                print(f'Piece count (white / black): {get_board_piece_count(agent.chess_game_agent.board)}')
                print(agent.chess_game_agent.board)
                # if (agent.move_counter.count % 50) == 0:
                #    draw_board(agent.board, display=True, verbosity=True)
            # Append the training data
            states.append(agent.chess_game_agent.board.fen())
            policy_targets.append(policy_target)

            old_used_nodes = agent.tree.get_list_of_all_used_nodes()
            agent.update_root(uci_move)
            agent.move_counter.increment()

            # delete unused nodes
            new_used_nodes = agent.tree.get_list_of_all_used_nodes()
            print(f'Old used nodes: {len(old_used_nodes)}')
            print(f'New used nodes: {len(new_used_nodes)}')
            nodes_to_delete = [node for node in old_used_nodes if node not in new_used_nodes]
            cnt = 0
            for node in nodes_to_delete:
                agent.tree.delete_node(node)
                cnt += 1

            print(f'Deleted {cnt} nodes')
            del nodes_to_delete, old_used_nodes, new_used_nodes
            gc.collect()

            if agent.move_counter.count > 3:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                print("[ Top 10 ]")
                for stat in top_stats[:10]:
                    print(stat)

            if agent.move_counter.count > 5:
                tracemalloc.stop()

            # Print the result of the game
            if agent.game_over() or agent.move_counter.count > config.maximum_moves:
                print(f'Game Over! Winner is {agent.chess_game_agent.board.result()}')
                if agent.move_counter.count > config.maximum_moves:
                    game_limit_stop = True
                # add the value outcomes to the training data
                value_target = None

                if agent.chess_game_agent.board.result() == '1-0':
                    value_target = 1
                elif agent.chess_game_agent.board.result() == '0-1':
                    value_target = -1
                elif agent.chess_game_agent.board.result() == '1/2-1/2':
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
                    state = board_to_input(config, states[j])
                    key_dict['state'] = state
                    key_dict['policy_target'] = policy_targets[j]
                    key_dict['value_target'] = value_targets[j]

                    # Save the training data
                    save_training_data(agent, key_id, key_dict)

        agent.tree = None
        agent = None
        del agent
        gc.collect()

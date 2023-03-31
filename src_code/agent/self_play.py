import gc
from config.config import Config
from src_code.agent.agent import AlphaZeroChess
from src_code.agent.agent import board_to_input, draw_board
from src_code.agent.utils import get_board_piece_count, save_training_data, get_var_sizes, print_variable_sizes_pympler


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

        key_id_list = []
        states = []
        policy_targets = []
        game_limit_stop = False
        # training loop:
        while not agent.game_over() and not game_limit_stop:
            # Get the current state of the board
            player = 'white' if agent.board.turn else 'black'
            uci_move, policy, policy_target = agent.get_action()

            # agent.tree.root.count_nodes()
            # print(f'Global variables: {print_variable_sizes_pympler(globals())}')
            # print(f'Local variables: {print_variable_sizes_pympler(locals())}')

            # Take the action and update the board state
            # print(uci_move)
            # if player == 'black':
            #     uci_move = input()

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
            gc.collect()
            agent.move_counter.increment()

            # Print the result of the game
            if agent.game_over() or agent.move_counter.count > config.maximum_moves:
                print(f'Game Over! Winner is {agent.board.result()}')
                if agent.move_counter.count > config.maximum_moves:
                    game_limit_stop = True
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

        agent.tree = None
        agent = None
        gc.collect()

import gc
from config.config import Config
from src_code.agent.agent import AlphaZeroChess, Node
from src_code.agent.agent import board_to_input, draw_board
from src_code.agent.utils import get_board_piece_count, save_training_data, get_var_sizes, \
    malloc_trim, print_variable_sizes_pympler, get_size, input_to_board
from src_code.in_out.windows import get_input, print_output


def play_game():
    # Initialize the config and agent
    config = Config(num_iterations=20, verbosity=True)
    # Play the game
    agent = AlphaZeroChess(config)
    agent.load_networks('network_current')
    iters = None
    iters_choices = [None, 40, 60, 100, 200, 400, 800, 1200]
    while not agent.game_over():
        # Get the current state of the board
        print(f'player to move: {agent.tree.root.player_to_move}')
        uci_move, policy, policy_target = agent.get_action(iters=iters)
        legal_moves = [str(x) for x in agent.board.legal_moves]

        print(f'_____ Statistics _________')
        agent.tree.width()
        print(f'Piece count (white / black): {get_board_piece_count(agent.board)}')
        Node.gather_statistics()
        print(f'_____ End Statistics -----')
        print(f'recommended move: {uci_move}')
        move = None
        while move is None:
            # Take the action and update the board state
            print(agent.tree.root.board)
            print(f'input the move for {agent.tree.root.player_to_move}: ')
            ret_values = get_input(f'Input the move for {agent.tree.root.player_to_move}:',
                                            'Select iterations:', legal_moves, [20, 40, 60, 100, 200, 400, 800, 1200],
                                            20)
            if ret_values is not None:
                move = ret_values[0]
                iters = int(ret_values[1])

        uci_move = move
        agent.board.push_uci(uci_move)
        print(agent.board)

        if agent.move_counter.count > agent.temperature_threshold:
            print(f'Updating temperature..')
            agent.update_temperature()

        # Update the tree
        old_node_list = agent.tree.root.get_all_nodes()
        agent.tree.update_root(uci_move)
        new_node_list = agent.tree.root.get_all_nodes()
        agent.tree.root.count_nodes()

        for abandoned_node in set(old_node_list).difference(set(new_node_list)):
            abandoned_node.remove_from_all_nodes()
            del abandoned_node

        del old_node_list, new_node_list

        agent.move_counter.increment()

        # Print the result of the game
        if agent.game_over() or agent.move_counter.count > config.maximum_moves:
            print(f'Game Over! Winner is {agent.board.result()}')

    agent.tree = None
    agent = None
    gc.collect()


if __name__ == '__main__':
    play_game()

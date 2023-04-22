import gc
from config.config import Config
from src_code.agent.agent import AlphaZeroChess, Node
from src_code.agent.agent import board_to_input, draw_board
from src_code.agent.utils import get_board_piece_count, save_training_data, get_var_sizes, \
    malloc_trim, print_variable_sizes_pympler, get_size, input_to_board


def play_game():
    # Initialize the config and agent
    config = Config(num_iterations=1200, verbosity=True)
    # Play the game
    agent = AlphaZeroChess(config)

    while not agent.game_over():
        # Get the current state of the board
        print(f'player to move: {agent.tree.root.player_to_move}')
        uci_move, policy, policy_target = agent.get_action()
        legal_moves = [x for x in agent.board.legal_moves]
        print(f'recommended move: {uci_move}')
        print(f'_____ Statistics _________')
        # insert mcts data
        print(f'_____ End Statistics -----')

        move = None
        while move is None:
            # Take the action and update the board state
            print(f'input the move for {agent.tree.root.player_to_move}: ')
            move_inp = input()

            cwc = 0
        # test for legal moves
        agent.board.push_uci(uci_move)

        # Print the board
        print(f'The {agent.move_counter.count} move was: {uci_move}')
        if agent.move_counter.count > agent.temperature_threshold:
            agent.update_temperature()
        if (agent.move_counter.count % 1) == 0 and (agent.move_counter.count > 0):
            # agent.tree.width()
            print(f'Piece count (white / black): {get_board_piece_count(agent.board)}')
            print(agent.board)
            # if (agent.move_counter.count % 50) == 0:
            #    draw_board(agent.board, display=True, verbosity=True)

        # Update the tree
        old_node_list = agent.tree.root.get_all_nodes()
        agent.tree.update_root(uci_move)
        new_node_list = agent.tree.root.get_all_nodes()
        agent.tree.root.count_nodes()

        for abandoned_node in set(old_node_list).difference(set(new_node_list)):
            abandoned_node.remove_from_all_nodes()
            del abandoned_node

        del old_node_list, new_node_list

        # objgraph.show_refs(agent.tree.root, filename=f'/home/cooneycw/root_refs.png')
        # objgraph.show_refs(agent.tree.network, filename=f'/home/cooneycw/network_refs.png')
        # objgraph.show_backrefs(agent.tree.root, filename=f'/home/cooneycw/root_backrefs.png')
        # objgraph.show_backrefs(agent.tree.network, filename=f'/home/cooneycw/network_backrefs.png')

        # objects = gc.get_objects()
        # print(f'Objects: {len(objects)}')
        #
        # # create a list of tuples containing each object and its size
        # obj_sizes = [(obj, sys.getsizeof(obj)) for obj in objects]
        #
        # # sort the list of tuples by the size of the objects
        # obj_sizes.sort(key=lambda x: x[1], reverse=True)
        #
        # # display the top 100 objects by size
        # for obj, size in obj_sizes[:100]:
        #     print(type(obj), size)
        #
        # list_objects = [obj for obj in gc.get_objects() if isinstance(obj, list)]
        # node_objects = [obj for obj in gc.get_objects() if isinstance(obj, Node)]
        #
        # print(f'Number of lists: {len(list_objects)}')
        # print(f'Number of nodes: {len(node_objects)}')
        #
        # del list_objects, node_objects, objects, size, obj_sizes, obj
        gc.collect()
        malloc_trim()
        agent.move_counter.increment()

        # Print the result of the game
        if agent.game_over() or agent.move_counter.count > config.maximum_moves:
            print(f'Game Over! Winner is {agent.board.result()}')

    agent.tree = None
    agent = None
    gc.collect()

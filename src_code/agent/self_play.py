import gc
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# from line_profiler_pycharm import profile
from collections import Counter
from config.config import Config
from src_code.agent.agent import AlphaZeroChess, Node
from src_code.agent.agent import board_to_input, draw_board
from src_code.agent.utils import get_board_piece_count, save_training_data, get_var_sizes, \
    malloc_trim, print_variable_sizes_pympler, get_size, input_to_board

tf.get_logger().setLevel('ERROR')
physical_devices = tf.config.list_physical_devices('GPU')
if len(tf.config.list_physical_devices('GPU')) > 0:
    gpu_idx = 0  # Set the index of the GPU you want to use
    # Get the GPU device
    gpu_device = physical_devices[gpu_idx]
    # Set the GPU memory growth
    tf.config.experimental.set_memory_growth(gpu_device, True)
else:
    print('No GPUs available')


# @profile
def play_games(pass_dict):
    game_id = pass_dict['game_id']
    verbosity = pass_dict['verbosity']
    learning_rate = pass_dict['learning_rate']
    network_name = pass_dict['network_name']
    run_type = pass_dict['run_type']
    pre_play = pass_dict['pre_play']
    config = Config(verbosity=verbosity)

    # Play the game
    # Initialize the agent
    agent = AlphaZeroChess(config)
    agent.load_networks(network_name)
    key_id_list = []
    states = []
    moves = []
    policy_targets = []
    game_limit_stop = False
    # training loop:
    while not agent.game_over() and not game_limit_stop:
        # Get the current state of the board
        player = 'white' if agent.board.turn else 'black'
        uci_move, policy, policy_target = agent.get_action(pre_play=pre_play)

        if run_type != 'ray':
            # Collect all prior values and node names from nodes
            prior_values = [(node.name, node.prior_prob, node.Qreward, node.Nvisit) for node in
                            list(agent.tree.root.children)]

            # Sort the prior values in ascending order
            sorted_values = sorted(prior_values, key=lambda x: x[2])

            # Display the bottom 10 nodes (lowest prior values)
            print("Bottom 15 nodes:")
            for name, prob, qreward, nvisit in sorted_values[:15]:
                print(f"Node: {name} Prior Prob: {prob:.4f} Qreward: {qreward:.4f} Nvisits: {nvisit}")

            # Display the top 10 nodes (highest prior values)
            print("\nTop 15 nodes:")
            for name, prob, qreward, nvisit in sorted_values[-15:]:
                print(f"Node: {name} Prior Prob: {prob:.4f} Qreward: {qreward:.4f} Nvisits: {nvisit}")

        agent.board.push_uci(uci_move)

        key_id = f'azChess_{game_id}_{agent.move_counter.count}_{datetime.datetime.now()}'
        key_id_list.append(key_id)
        # Print the board
        print(f'{network_name} - {agent.move_counter.count} move was: {uci_move}')
        if agent.move_counter.count > agent.temperature_threshold:
            agent.update_temperature()
        if (agent.move_counter.count % 1) == 0 and (agent.move_counter.count > 0):
            print(f'{network_name} - Piece count (white / black): {get_board_piece_count(agent.board)}')
            print(agent.board)
            # agent.tree.width()
            # agent.tree.gather_tree_statistics()
            # if (agent.move_counter.count % 50) == 0:
            #    draw_board(agent.board, display=True, verbosity=True)
        # Append the training data
        state = board_to_input(config, agent.tree.root)
        states.append(state)
        moves.append(uci_move)
        policy_targets.append(policy_target)

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
            if agent.move_counter.count > config.maximum_moves:
                game_limit_stop = True
            # add the value outcomes to the training data
            value_target = None

            if agent.board.result(claim_draw=True) == '1-0':
                value_target = 1
            elif agent.board.result(claim_draw=True) == '0-1':
                value_target = -1
            elif agent.board.result(claim_draw=True) == '1/2-1/2':
                # modify for white players
                if player == 'white':
                    value_target = 0
                else:
                    value_target = -0
            else:
                value_target = 0

            value_targets = [value_target * config.reward_discount ** (len(policy_targets) - (i+1)) for i in range(len(policy_targets))]

            # Update the game counter
            config.game_counter.increment()

            for j, key_id in enumerate(key_id_list):
                key_dict = dict()
                key_dict['key'] = key_id
                key_dict['game_id'] = game_id
                key_dict['move_id'] = j
                key_dict['state'] = states[j]
                key_dict['move'] = moves[j]
                key_dict['policy_target'] = policy_targets[j]
                key_dict['value_target'] = value_targets[j]

                # Save the training data
                save_training_data(agent, key_id, key_dict)

            states = []
            moves = []
            policy_targets = []
            value_targets = None
            key_dict = None

    agent.tree = None
    agent = None
    gc.collect()

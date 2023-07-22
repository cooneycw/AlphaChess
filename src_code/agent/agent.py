import gc
import os
import chess
import redis
import math
import copy
import random
import pickle
import numpy as np
import requests
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from collections import deque
# from line_profiler_pycharm import profile
from src_code.agent.utils import draw_board, malloc_trim
from src_code.agent.network import create_network


class AlphaZeroChess:
    def __init__(self, config, network=None):
        self.config = config
        self.board = chess.Board()
        # self.board = chess.Board(None)
        # # Place the black king at b8 (index 17)
        # self.board.set_piece_at(32, chess.Piece(chess.KING, chess.BLACK))
        #
        # # Place the white king at c6 (index 34)
        # self.board.set_piece_at(0, chess.Piece(chess.KING, chess.WHITE))
        #
        # # Place the white pawn at a2
        # self.board.set_piece_at(24, chess.Piece(chess.QUEEN, chess.BLACK))

        self.temperature = config.temperature
        self.min_temperature = config.min_temperature
        self.temperature_threshold = config.temperature_threshold
        self.sim_counter = config.SimCounter()
        self.move_counter = config.MoveCounter()
        self.redis = redis.StrictRedis(host=config.redis_host, port=config.redis_port, db=config.redis_db)
        # Create the network
        if network is None:
            self.network = create_network(config)
        else:
            self.network = network

        # Assign the optimizer to self.optimizer
        self.optimizer = config.optimizer

        # Initialize the MCTS tree
        self.tree = MCTSTree(self)

    def update_temperature(self):
        self.temperature = self.min_temperature

    def reset(self):
        self.board.reset()
        self.tree = MCTSTree(self)
        self.move_counter = self.config.MoveCounter()

    # @profile
    def get_action(self, iters=None, pre_play=False, eval=False):
        if iters is None:
            iters = self.config.num_iterations
        if eval is True:
            iters = self.config.eval_num_iterations
        if pre_play is True:
            iters = self.config.preplay_num_iterations

        """Get the best action to take given the current state of the board."""
        """Uses dirichlet noise to encourage exploration in place of temperature."""
        while self.sim_counter.get_count() < iters:
            _ = self.tree.process_mcts(self.tree.root, self.config, self.network, eval, pre_play)
            self.sim_counter.increment()
            if self.config.verbosity is True:
                if self.sim_counter.get_count() % 100 == 0:
                    print(f'Game Number: {self.config.game_counter.get_count()} Move Number: {self.move_counter.get_count()} Simulations: {self.sim_counter.get_count()}')
                    # self.tree.width()

        # retrieve the updated policy
        policy, policy_uci, temp_adj_policy, policy_array = self.tree.get_policy(self)
        print(f'max adj: {max(temp_adj_policy)}  sum_adj: {sum(temp_adj_policy)}  max_pol: {max(policy)}  sum_pol: {sum(policy)}  temp: {self.temperature}')
        comparator = np.random.rand()
        cumulative_prob = 0
        for i in range(len(temp_adj_policy)):  # len(temp_adj_policy)
            cumulative_prob += temp_adj_policy[i]
            if cumulative_prob > comparator:
                action = i
                break
            action = len(temp_adj_policy) - 1

        self.sim_counter.reset()
        return policy_uci[action], policy, policy_array

    def game_over(self):
        return self.board.is_game_over(claim_draw=True)

    def update_network(self, train_states, train_policy, train_value, val_states, val_policy, val_value):
        """Update the neural network with the latest training data."""
        # Split the data into training and validation sets
        train_dataset = self.config.ChessDataset(train_states, train_policy, train_value)
        train_dataloader = tf.data.Dataset.from_generator(lambda: train_dataset,
                                                          (tf.float64, tf.float64, tf.float64)).batch(
            self.config.batch_size)
        val_dataset = self.config.ChessDataset(val_states, val_policy, val_value)
        val_dataloader = tf.data.Dataset.from_generator(lambda: val_dataset,
                                                        (tf.float64, tf.float64, tf.float64)).batch(
            self.config.batch_size)
        validation_loss_tot = 0
        validation_loss_cnt = 0
        for epoch in range(self.config.num_epochs):
            avg_train_loss = 0
            avg_train_accuracy = 0
            avg_train_value_mae = 0
            num_train_batches = 0
            # Train the model using the training data
            for inputs, policy_targets, value_targets in train_dataloader:
                with tf.GradientTape() as tape:
                    value_targets = tf.expand_dims(value_targets, 1)
                    policy_preds, value_preds = self.network(inputs, training=True)

                    value_loss_per_instance = keras.losses.mean_squared_error(value_targets, value_preds)
                    value_loss = tf.reduce_mean(value_loss_per_instance)

                    policy_loss_per_instance = keras.losses.categorical_crossentropy(policy_targets, policy_preds)
                    policy_loss = tf.reduce_mean(policy_loss_per_instance)

                    loss = value_loss + policy_loss

                # Compute gradients
                gradients = tape.gradient(loss, self.network.trainable_variables)

                # Apply gradient clipping
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_gradient_norm)

                # Update the model parameters with clipped gradients
                self.optimizer.apply_gradients(zip(clipped_gradients, self.network.trainable_variables))

                avg_train_loss += loss
                value_mae = tf.reduce_mean(
                    tf.abs(tf.cast(value_targets, tf.float32) - tf.cast(value_preds, tf.float32)))
                avg_train_value_mae += value_mae.numpy()
                policy_accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(policy_targets, axis=1), tf.argmax(policy_preds, axis=1)), tf.float64))
                avg_train_accuracy += policy_accuracy.numpy()
                num_train_batches += 1

            avg_train_loss /= num_train_batches
            avg_train_accuracy /= num_train_batches
            avg_train_value_mae /= num_train_batches

            # Evaluate the model on the validation data
            avg_val_loss = 0
            avg_val_accuracy = 0
            avg_val_value_mae = 0
            num_val_batches = 0
            for inputs, policy_targets, value_targets in val_dataloader:
                value_targets = tf.expand_dims(value_targets, 1)
                policy_preds, value_preds = self.network(inputs, training=False)

                value_loss_per_instance = keras.losses.mean_squared_error(value_targets, value_preds)
                value_loss = tf.reduce_mean(value_loss_per_instance)

                policy_loss_per_instance = keras.losses.categorical_crossentropy(policy_targets, policy_preds)
                policy_loss = tf.reduce_mean(policy_loss_per_instance)

                loss = value_loss + policy_loss

                avg_val_loss += loss
                policy_accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(policy_targets, axis=1), tf.argmax(policy_preds, axis=1)), tf.float64))
                avg_val_accuracy += policy_accuracy.numpy()
                value_mae = tf.reduce_mean(
                    tf.abs(tf.cast(value_targets, tf.float32) - tf.cast(value_preds, tf.float32)))
                avg_val_value_mae += value_mae.numpy()

                num_val_batches += 1
                validation_loss_tot += avg_val_loss
                validation_loss_cnt += num_val_batches

            avg_val_loss /= num_val_batches
            avg_val_accuracy /= num_val_batches
            avg_val_value_mae /= num_val_batches

            print(f'Epoch: {epoch+1} Train Batches: {num_train_batches}  Val Batches: {num_val_batches}')
            print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}')
            print(f'Train Value MAE: {avg_train_value_mae:.4f}, Val Value MAE: {avg_val_value_mae:.4f}')

        return validation_loss_tot, validation_loss_cnt

    def load_network_weights(self, key_name):
        # Connect to Redis and retrieve the serialized weights
        serialized_weights = self.redis.get(key_name)

        if serialized_weights is None:
            # Initialize the weights if no weights are found in Redis
            raise Exception(f'No weights found in Redis: {key_name}')

        else:
            # Deserialize the weights from the byte string using NumPy
            weights_dict = pickle.loads(serialized_weights)

            # Set the weights for each layer of the network
            for layer in self.network.layers:
                layer_name = layer.name
                if (layer_name[0:5] == 'input' or
                        layer_name[0:7] == 'res_add' or
                        layer_name[0:10] == 'value_relu' or
                        layer_name[0:11] == 'policy_relu' or
                        layer_name[0:11] == 'policy_conv' or
                        layer_name[0:9] == 'policy_bn' or
                        layer_name[0:8] == 'value_bn' or
                        layer_name[0:10] == 'value_conv' or
                        layer_name[0:13] == 'value_flatten' or
                        layer_name[0:13] == 'value_leakyre' or
                        layer_name[0:14] == 'policy_flatten' or
                        layer_name[0:14] == 'policy_leakyre' or
                        layer_name[0:10] == 'activation' or
                        layer_name[0:4] == 'relu' or
                        layer_name[0:7] == 'dropout' or
                        layer_name[0:5] == 'leaky' or
                        layer_name[0:9] == 'res_leaky' or
                        layer_name[0:8] == 'res_relu'):
                    continue
                layer_weights = weights_dict[layer_name]
                layer.set_weights([np.array(w) for w in layer_weights])

            print(f"Network weights loaded from Redis key '{key_name}'")

    def save_networks(self, key_name):
        self.save_network_weights(key_name)

    # def save_networks_tf(self):
    #     current_path = os.getcwd()
    #     model_path = os.path.join(current_path, 'model')
    #     self.network.save(model_path, 'network_for_conversion')

    def load_networks(self, key_name):
        self.load_network_weights(key_name)
        # self.save_networks_tf()
        #
        # # Convert the model to TensorRT
        # conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        # conversion_params = conversion_params._replace(max_workspace_size_bytes=(1 << 32))
        # conversion_params = conversion_params._replace(precision_mode="FP16")
        # conversion_params = conversion_params._replace(maximum_cached_engines=100)
        #
        # current_path = os.getcwd()
        # model_path = os.path.join(current_path, 'model')
        #
        # converter = trt.TrtGraphConverterV2(
        #     input_saved_model_dir=model_path,  # you should save your model first
        #     conversion_params=conversion_params
        # )
        #
        # converter.convert()
        #
        # def input_fn():
        #     input_shapes = [1, self.config.board_size, self.config.board_size,
        #                     self.config.num_channels]  # replace with your input shape
        #     yield [np.random.normal(size=input_shapes).astype(np.float32)]
        #
        # converter.build(input_fn)
        #
        # # The converted function that is used for inference
        # self.network = converter.converted_call

    def save_network_weights(self, key_name):
        # Convert the weights to a dictionary

        weights_dict = {}
        for layer in self.network.layers:
            if len(layer.get_weights()) > 0:  # Check if the layer has any weights
                weights_dict[layer.name] = [w.tolist() for w in layer.get_weights()]

        # Serialize the dictionary to a byte string using NumPy
        pickle_dict = pickle.dumps(weights_dict)
        # Connect to Redis and save the weights using the specified key name
        self.redis.set(key_name, pickle_dict)

        print(f"Network weights saved to Redis key {key_name}")


def board_to_input_single(config, node):
    # Create an empty 8x8x119 tensor
    piece_map = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
    # input_tensor = np.zeros((config.board_size, config.board_size, config.num_channels))
    input_tensor = np.zeros((config.board_size, config.board_size, 14), dtype=np.float32)

    board_ind = 0
    curr_board = None
    curr_board = node.board.copy()

    # Encode the piece positions in channels 1-12
    for square, piece in curr_board.piece_map().items():
        if piece.color == chess.WHITE:
            piece_idx = piece.piece_type - 1
        else:
            piece_idx = piece.piece_type - 1 + 6
        input_tensor[chess.square_rank(square), chess.square_file(square), (board_ind * 14) + piece_idx] = 1

    # repetitions
    last_move = node.prior_moves[board_ind]
    last_move_1 = node.prior_moves[board_ind + 2]
    last_move_2 = node.prior_moves[board_ind + 4]
    opp_last_move = node.prior_moves[board_ind + 1]
    opp_last_move_1 = node.prior_moves[board_ind + 3]
    opp_last_move_2 = node.prior_moves[board_ind + 5]

    reps = 0
    opp_reps = 0
    if last_move is None:
        pass
    elif last_move == last_move_1:
        if last_move_1 == last_move_2:
            reps = 2 / 2
        else:
            reps = 1 / 2

    if opp_last_move is None:
        pass
    elif opp_last_move == opp_last_move_1:
        if opp_last_move_1 == opp_last_move_2:
            opp_reps = 2 / 2
        else:
            opp_reps = 1 / 2

    if curr_board.turn is True:
        input_tensor[:, :, (board_ind * 14) + 12] = reps
        input_tensor[:, :, (board_ind * 14) + 13] = opp_reps
    else:
        input_tensor[:, :, (board_ind * 14) + 12] = opp_reps
        input_tensor[:, :, (board_ind * 14) + 13] = reps

    return input_tensor


def board_to_input(config, node):
    # Create an empty 8x8x119 tensor
    input_tensor = np.zeros((config.board_size, config.board_size, config.num_channels), dtype=np.float32)

    board_ind = 0
    curr_board = None
    while board_ind < 8:
        if board_ind == 0:
            single_tensor = board_to_input_single(config, node)
            input_tensor[:, :, 0:14] = single_tensor
        else:
            if node.prior_boards[board_ind - 1] is None:
                break
            else:
                inds = np.arange((board_ind * 14), (14 * board_ind) + 14)
                input_tensor[:, :, inds] = node.prior_boards[board_ind - 1]

        # Encode the piece positions in channels 1-12

        board_ind += 1

    # Encode the current player in the first channel
    input_tensor[:, :, 112] = (node.board.turn * 1.0)

    # Encode the fullmove number in channel 14
    input_tensor[:, :, 113] = node.board.fullmove_number / config.maximum_moves

    if node.prior_moves[0] is not None:
        is_white_kingside_castle = node.prior_moves[0] == chess.Move.from_uci("e1g1")
        is_white_queenside_castle = node.prior_moves[0] == chess.Move.from_uci("e1c1")
        is_black_kingside_castle = node.prior_moves[0] == chess.Move.from_uci("e8g8")
        is_black_queenside_castle = node.prior_moves[0] == chess.Move.from_uci("e8c8")

        if is_white_kingside_castle:
            input_tensor[:, :, 114] = 1
        if is_white_queenside_castle:
            input_tensor[:, :, 115] = 1
        if is_black_kingside_castle:
            input_tensor[:, :, 116] = 1
        if is_black_queenside_castle:
            input_tensor[:, :, 117] = 1

    # Encode the "no progress" count in channel 118
    input_tensor[:, :, 118] = node.board.halfmove_clock / config.maximum_moves

    return input_tensor


def get_legal_moves(board):
    """
    Return a list of all legal moves for the current player on the given board.
    """
    legal_moves = list(board.legal_moves)
    del board
    return [move.uci() for move in legal_moves]


def move_to_index(move):
    # Convert the move string to a chess.Move object
    move_obj = chess.Move.from_uci(move)
    # Convert the move object to an integer index
    index = move_obj.from_square * 64 + move_obj.to_square
    return index


class MCTSTree:
    # https://towardsdatascience.com/monte-carlo-tree-search-an-introduction-503d8c04e168
    def __init__(self, az):
        self.root = Node(az.board.copy())
        self.config = az.config

    def get_policy(self, agent):
        # Get the policy from the root node
        epsilon = 1e-8
        policy = [child.Nvisit for child in self.root.children]
        policy_uci = [child.name for child in self.root.children]

        if any(math.isnan(pol) for pol in policy):
            policy = np.array([1 * self.root.children[i].board.is_game_over(claim_draw=True) for i in range(len(self.root.children))])
        # Normalize the policy
        policy = np.array(policy) / (sum(policy) + epsilon)

        # Adjust the policy according to the temperature
        temp_adj_policy = np.power(policy, 1 / agent.temperature)
        temp_adj_policy /= np.sum(np.power(policy, 1 / agent.temperature)) + epsilon

        policy_array = policy_to_prob_array(policy, policy_uci,
                                            self.config.all_chess_moves)

        return policy, policy_uci, temp_adj_policy, policy_array

    # @profile
    def process_mcts(self, node, config, network, eval, pre_play):
        if eval is True:
            c_puct = self.config.eval_c_puct
        else:
            c_puct = self.config.c_puct
        epsilon = 1e-8

        if node.board.is_game_over(claim_draw=True):
            winner = node.board.result()
            if winner == '1-0':
                v = 1
            elif winner == '0-1':
                v = -1
            elif winner == '1/2-1/2' or winner == '*':
                if node.player_to_move == 'white':
                    v = -0
                elif node.player_to_move == 'black':
                    v = 0
            else:
                print(f'Here is an unknown code: {winner}')
                v = 0
            return v

        # Select a node to expand
        if len(node.children) == 0:
            v = self.expand(node, network, pre_play)
            return v

        # Evaluate the node
        max_uct = -float('inf')
        best_node = None

        adj = 1
        if node.player_to_move == 'black':
            adj = -1

        node_visits = 0
        for child in node.children:
            node_visits += child.Nvisit

        # if node_visits == 0:
        #    best_node = random.choice(node.children)
        #else:
        indices = list(range(len(node.children)))
        random.shuffle(indices)
        for index in indices:
            child = node.children[index]
            # adj only applied to Qreward (not probabilities)
            uct = (child.Qreward * adj) + c_puct * child.prior_prob * math.sqrt(
                node_visits + epsilon) / (1 + child.Nvisit)

            if uct > max_uct:
                max_uct = uct
                best_node = child

        # Simulate a game from the best_node
        v = self.process_mcts(best_node, config, network, eval, pre_play)

        # Backpropagate the results of the simulation
        best_node.Qreward = (best_node.Qreward * best_node.Nvisit + v) / (best_node.Nvisit + 1)
        best_node.Nvisit += 1

        return v

    # @profile
    def expand(self, leaf_node, network, pre_play):
        # Get the policy and value from the neural network
        state = board_to_input(self.config, leaf_node)
        # state_list = state.tolist()  # Convert numpy array to list
        # response = requests.post('http://192.168.5.133:8000/predict', json={'state': state_list})
        # Check the response status code to ensure the request was successful
        #if response.status_code == 200:
        #    preds = response.json()
        #    pi = np.array(preds['policy_preds'])
        #    v = np.array(preds['value_preds'])
        #    leaf_node.prior_value = v[0][0]
        #    del v
        #else:
        #    print(f'Request failed with status code {response.status_code}')

        #  old_code Create a TensorFlow dataset using the expanded input
        state_expanded = np.expand_dims(state, axis=0)
        state_ds = tf.data.Dataset.from_tensor_slices(state_expanded)
        state_ds_batched = state_ds.batch(1)
        pi, v = network.predict(state_ds_batched, verbose=0)
        if pre_play is False:
            v = v[0][0]
        else:
            v = 0

        # Add Dirichlet noise to the prior probabilities
        if leaf_node.name == 'root':
            alpha = self.config.dirichlet_alpha
            noise = np.random.dirichlet(alpha * np.ones(len(pi[0])))
            pi = (1 - self.config.eps) * pi[0] + self.config.eps * noise
            pi = np.array(pi) / sum(pi)
        else:
            pi = pi[0]
            pi = np.array(pi) / sum(pi)

        legal_moves = get_legal_moves(leaf_node.board.copy())

        # Create list of legal policy probabilities corresponding to legal moves
        epsilon = 1e-8
        ep_threshold = 20 * epsilon

        legal_probabilities = np.array([pi[self.config.all_chess_moves.index(move)] for move in legal_moves],
                                       dtype=np.float64)
        if sum(legal_probabilities) < ep_threshold:
            legal_probabilities = np.ones(len(legal_moves)) / len(legal_moves)
        else:
            legal_probabilities /= (np.sum(legal_probabilities) + epsilon)

        del pi

        # Normalize the legal probabilities to sum to 1

        for i, action in enumerate(legal_moves):
            new_prior_moves = copy.deepcopy(leaf_node.prior_moves)
            new_prior_boards = copy.deepcopy(leaf_node.prior_boards)
            _ = new_prior_moves.pop()
            _ = new_prior_boards.pop()
            new_prior_moves.insert(0, action)
            new_prior_boards.insert(0, board_to_input_single(self.config, leaf_node))
            new_board = leaf_node.board.copy()
            new_board.push_uci(action)
            if leaf_node.player_to_move == 'white':
                player_to_move = 'black'
            else:
                player_to_move = 'white'
            child = Node(new_board.copy(), name=action, player_to_move=player_to_move, prior_boards=new_prior_boards, prior_moves=new_prior_moves)
            child.set_parent(leaf_node)

            child.prior_prob = legal_probabilities[i]
            leaf_node.children.append(child)

        del new_board, state, legal_moves, new_prior_boards, new_prior_moves
        return v

    def update_root(self, action):
        for child in self.root.children:
            if child.name == action:
                self.root = child
                self.root.parent = None
                self.root.name = 'root'
                break

    # def remove_node_and_descendants(self, node):
    #     for child in node.children:
    #         self.remove_node_and_descendants(child)
    #         # remove the child node from the tree
    #         if child in node.children:
    #             node.children.remove(child)
    #             child = None
    #             # print(f"Removed child node {child.name} from parent node {node.name}")
    #         else:
    #             pass
    #             # print(f"Child node {child.name} not found in parent node {node.name}")
    #         del child
    #     # remove the current node from the tree
    #     if node.parent is not None and node in node.parent.children:
    #         node.parent.children.remove(node)
    #         node = None
    #         # print(f"Removed node {node.name} from parent node {node.parent.name}")
    #     else:
    #         pass
    #         # print(f"Node {node.name} not found in parent node {node.parent.name}")
    #     del node

    def depth(self):
        print("Calculating depth...")
        depth = self._depth(self.root)
        print("Depth:", depth)
        return depth

    def _depth(self, node):
        if not node.children:
            return 0
        else:
            return 1 + max(self._depth(child) for child in node.children)

    def width(self):
        print("Calculating width...")
        node_counts = []
        self._width(self.root, node_counts, 0)
        print("Width:", node_counts)
        return node_counts

    def _width(self, node, node_counts, depth):
        if depth == len(node_counts):
            node_counts.append(1)
        else:
            node_counts[depth] += 1
        for child in node.children:
            self._width(child, node_counts, depth + 1)

    def gather_tree_statistics(self, exploration_factor=0):
        if not self.root:
            print("No nodes to gather statistics from.")
            return

        all_nodes = self.root.get_all_nodes()

        game_over_nodes = [node for node in all_nodes if node.board.is_game_over(claim_draw=True)]
        shortest_paths = self.find_shortest_paths_to_game_over_by_tree()

        most_visited_nodes = sorted(all_nodes, key=lambda node: node.Nvisit, reverse=True)[:5]
        least_visited_nodes = sorted(all_nodes, key=lambda node: node.Nvisit)[:5]

        highest_qreward_nodes = sorted(all_nodes, key=lambda node: node.Qreward, reverse=True)[:5]
        lowest_qreward_nodes = sorted(all_nodes, key=lambda node: node.Qreward)[:5]

        print(f"\nNumber of Game Over Nodes: {len(game_over_nodes)}")

        print("\nShortest Paths to Game Over:")
        for path in shortest_paths[:3]:
            print(" -> ".join(node.name for node in path))

        print("\nMost Visited Nodes:")
        for node in most_visited_nodes:
            print(f"Node: {node.path_from_root()}, Nvisit: {node.Nvisit}")

        print("\nLeast Visited Nodes:")
        for node in least_visited_nodes:
            print(f"Node: {node.path_from_root()}, Nvisit: {node.Nvisit}")

        print("\nHighest Qreward Nodes:")
        for node in highest_qreward_nodes:
            print(f"Node: {node.path_from_root()}, Qreward: {node.Qreward:.3f}")

        print("\nLowest Qreward Nodes:")
        for node in lowest_qreward_nodes:
            print(f"Node: {node.path_from_root()}, Qreward: {node.Qreward:.3f}")

    def find_shortest_paths_to_game_over_by_tree(self):
        game_over_nodes = [node for node in self.root.get_all_nodes() if node.board.is_game_over(claim_draw=True)]

        if not game_over_nodes:
            return []

        shortest_paths = []
        for game_over_node in game_over_nodes:
            path = deque()
            current_node = game_over_node
            while current_node:
                path.appendleft(current_node)
                current_node = current_node.parent
            shortest_paths.append(list(path))

        shortest_paths.sort(key=lambda path: len(path))
        return shortest_paths


def policy_to_prob_array(policy, legal_moves, all_moves_list):
    prob_array = np.zeros(len(all_moves_list))

    for i, move in enumerate(all_moves_list):
        if move in legal_moves:
            index = legal_moves.index(move)
            prob_array[i] = policy[index]

    return prob_array


class Node:
    __slots__ = 'board', 'Qreward', 'Nvisit', 'prior_prob', 'children', 'player_to_move', \
                'parent', 'name', 'prior_boards', 'prior_moves'
    all_nodes = set()

    def __init__(self, board, player_to_move='white', name='root', prior_boards=None, prior_moves=None):
        self.board = board
        self.Qreward = 0
        self.Nvisit = 0
        self.prior_prob = 0
        self.children = []
        self.player_to_move = player_to_move
        self.parent = None
        self.name = name
        if prior_boards is None:
            self.prior_boards = [None] * 7
            self.prior_moves = [None] * (8 + 5)
        else:
            self.prior_boards = prior_boards
            self.prior_moves = prior_moves
        Node.all_nodes.add(self)

    def set_parent(self, parent):
        self.parent = parent

    def remove_from_all_nodes(self):
        del self.board
        del self.children
        del self.parent
        del self.Qreward
        del self.Nvisit
        del self.prior_prob
        del self.name
        del self.player_to_move
        del self.prior_boards
        del self.prior_moves
        Node.all_nodes.discard(self)

    def get_all_nodes(self):
        all_nodes = [self]
        for child in self.children:
            all_nodes += child.get_all_nodes()
        return all_nodes

    def count_nodes(self):
        # Recursively traverse the tree and increment the counter for each node
        count = 1  # Count the current node
        for child in self.children:
            count += child.count_nodes()[0]
        return count, len(Node.all_nodes)

    def count_lists(self):
        count = 0
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, list):
                count += 1
        for child in self.children:
            count += child.count_lists()
        return count

    def ucb1(self, parent_Nvisit, exploration_factor):
        epsilon = 1e-6
        if self.Nvisit == 0 or parent_Nvisit == 0:
            return float('inf')
        exploitation_term = self.Qreward / self.Nvisit
        exploration_term = math.sqrt(math.log(parent_Nvisit) / self.Nvisit)

        return exploitation_term + exploration_factor * exploration_term

    def path_from_root(self):
        path = []
        current_node = self
        while current_node is not None:
            path.append(current_node.name)
            current_node = current_node.parent
        return " -> ".join(reversed(path))

    @classmethod
    def gather_statistics(cls, exploration_factor=0):
        if not cls.all_nodes:
            print("No nodes to gather statistics from.")
            return

        ucb1_values = {node: node.ucb1(node.parent.Nvisit if node.parent else 1, exploration_factor) for node in
                       cls.all_nodes}
        sorted_nodes = sorted(ucb1_values.items(), key=lambda item: item[1], reverse=True)

        most_valuable_nodes = sorted_nodes[:5]  # Top 5 nodes
        least_valuable_nodes = sorted_nodes[-5:]  # Bottom 5 nodes

        game_over_nodes = [node for node in cls.all_nodes if node.board.is_game_over(claim_draw=True)]
        shortest_paths = cls.find_shortest_paths_to_game_over()

        most_visited_nodes = sorted(cls.all_nodes, key=lambda node: node.Nvisit, reverse=True)[:5]
        least_visited_nodes = sorted(cls.all_nodes, key=lambda node: node.Nvisit)[:5]

        highest_prior_value_nodes = sorted(cls.all_nodes, key=lambda node: node.prior_value, reverse=True)[:5]
        lowest_prior_value_nodes = sorted(cls.all_nodes, key=lambda node: node.prior_value)[:5]

        highest_qreward_nodes = sorted(cls.all_nodes, key=lambda node: node.Qreward, reverse=True)[:5]
        lowest_qreward_nodes = sorted(cls.all_nodes, key=lambda node: node.Qreward)[:5]

        print("\nHighest Prior Value Nodes:")
        for node in highest_prior_value_nodes:
            print(f"Node: {node.path_from_root()}, Prior Value: {node.prior_value:.3f}")

        print("\nLowest Prior Value Nodes:")
        for node in lowest_prior_value_nodes:
            print(f"Node: {node.path_from_root()}, Prior Value: {node.prior_value:.3f}")

        # print("\nMost Valuable Nodes:")
        # for node, ucb1_value in most_valuable_nodes:
        #     print(f"Node: {node.path_from_root()}, UCB1 Value: {ucb1_value:.3f}")
        #
        # print("\nLeast Valuable Nodes:")
        # for node, ucb1_value in least_valuable_nodes:
        #     print(f"Node: {node.path_from_root()}, UCB1 Value: {ucb1_value:.3f}")

        print(f"\nNumber of Game Over Nodes: {len(game_over_nodes)}")

        print("\nShortest Paths to Game Over:")
        for path in shortest_paths[:3]:
            print(" -> ".join(node.name for node in path))

        print("\nMost Visited Nodes:")
        for node in most_visited_nodes:
            print(f"Node: {node.path_from_root()}, Nvisit: {node.Nvisit}")

        print("\nLeast Visited Nodes:")
        for node in least_visited_nodes:
            print(f"Node: {node.path_from_root()}, Nvisit: {node.Nvisit}")

        print("\nHighest Qreward Nodes:")
        for node in highest_qreward_nodes:
            print(f"Node: {node.path_from_root()}, Qreward: {node.Qreward:.3f}")

        print("\nLowest Qreward Nodes:")
        for node in lowest_qreward_nodes:
            print(f"Node: {node.path_from_root()}, Qreward: {node.Qreward:.3f}")

    @classmethod
    def find_shortest_paths_to_game_over(cls):
        game_over_nodes = [node for node in cls.all_nodes if node.board.is_game_over(claim_draw=True)]

        if not game_over_nodes:
            return []

        shortest_paths = []
        for game_over_node in game_over_nodes:
            path = deque()
            current_node = game_over_node
            while current_node:
                path.appendleft(current_node)
                current_node = current_node.parent
            shortest_paths.append(list(path))

        shortest_paths.sort(key=lambda path: len(path))
        return shortest_paths

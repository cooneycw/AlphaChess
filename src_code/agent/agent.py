import redis
import math
import tracemalloc
import copy
import gc
import weakref
import random
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from src_code.agent.utils import draw_board
from src_code.agent.chess_env import ChessGame
from src_code.agent.network import create_network


class AlphaZeroChess:
    def __init__(self, config, network=None):
        self.config = config
        self.chess_game_agent = ChessGame()
        # # Place the black king at b8 (index 17)
        # self.board.set_piece_at(32, chess.Piece(chess.KING, chess.BLACK))
        #
        # # Place the white king at c6 (index 34)
        # self.board.set_piece_at(0, chess.Piece(chess.KING, chess.WHITE))
        #
        # # Place the white pawn at a2
        # self.board.set_piece_at(24, chess.Piece(chess.QUEEN, chess.BLACK))

        self.num_channels = 17
        self.num_moves = 64 * 64
        self.temperature = config.temperature
        self.min_temperature = config.min_temperature
        self.temperature_threshold = config.temperature_threshold
        self.sim_counter = config.SimCounter()
        self.move_counter = config.MoveCounter()
        self.redis = redis.StrictRedis(host=config.redis_host, port=config.redis_port, db=config.redis_db)
        # Create the value networks
        if network is None:
            self.network = create_network(config)
            self.load_network_weights('network_current')
        else:
            self.network = network

        # Assign the optimizer to self.optimizer
        self.optimizer = config.optimizer

        # Initialize the MCTS tree
        self.tree = MCTSTree(self)

    def update_temperature(self):
        self.temperature = self.min_temperature

    def reset(self):
        self.tree = MCTSTree(self)
        self.move_counter = self.config.MoveCounter()

    def get_action(self, iters=None):
        if iters is None:
            iters = self.config.num_iterations

        """Get the best action to take given the current state of the board."""
        """Uses dirichlet noise to encourage exploration in place of temperature."""
        while self.sim_counter.get_count() < iters:
            first_expand = True
            _ = self.tree.process_mcts(self.tree.root, self.config, first_expand)
            self.sim_counter.increment()
            if self.sim_counter.get_count() % int(0.5 + 0.5*self.config.num_iterations) == 0:
                print(f'Game Number: {self.config.game_counter.get_count()} Move Number: {self.move_counter.get_count()} Number of simulations: {self.sim_counter.get_count()}')
                self.tree.width()

        # retrieve the updated policy
        if self.tree.root.player_to_move == 'white':
            policy, policy_uci, temp_adj_policy, policy_array = self.tree.get_policy_white(self)
        else:
            policy, policy_uci, temp_adj_policy, policy_array = self.tree.get_policy_black(self)

        comparator = np.random.rand()
        cumulative_prob = 0
        for i in range(len(temp_adj_policy)):  # len(temp_adj_policy)
            cumulative_prob += temp_adj_policy[i]
            if cumulative_prob > comparator:
                action = i
                break
            action = len(temp_adj_policy) - 1

        self.sim_counter.reset()
        del temp_adj_policy, comparator
        gc.collect()
        return policy_uci[action], policy, policy_array

    def game_over(self):
        return self.chess_game_agent.board.is_game_over(claim_draw=True)

    def update_root(self, uci_move):
        for child in self.tree.root.children:
            if child.name == uci_move:
                self.tree.delete_node(self.tree.root)
                self.tree.root = child
                self.tree.root.parent = None
                self.tree.root.name = 'root'
                break

    def update_network(self, states, policy_targets, value_targets):
        """Update the neural network with the latest training data."""
        # Split the data into training and validation sets
        train_states, val_states, train_policy, val_policy, train_value, val_value = train_test_split(
            states, policy_targets, value_targets, test_size=self.config.validation_split)

        train_dataset = self.config.ChessDataset(train_states, train_policy, train_value)
        train_dataloader = tf.data.Dataset.from_generator(lambda: train_dataset,
                                                          (tf.float32, tf.float32, tf.float32)).batch(
            self.config.batch_size)

        val_dataset = self.config.ChessDataset(val_states, val_policy, val_value)
        val_dataloader = tf.data.Dataset.from_generator(lambda: val_dataset,
                                                        (tf.float32, tf.float32, tf.float32)).batch(
            self.config.batch_size)

        for epoch in range(self.config.num_epochs):
            avg_train_loss = 0
            avg_train_accuracy = 0
            num_train_batches = 0

            # Train the model using the training data
            for inputs, policy_targets, value_targets in train_dataloader:
                with tf.GradientTape() as tape:
                    policy_preds, value_preds = self.network(inputs, training=True)
                    value_loss = keras.losses.mean_squared_error(value_targets, value_preds)
                    policy_loss = keras.losses.categorical_crossentropy(policy_targets, policy_preds)
                    loss = value_loss + policy_loss
                gradients = tape.gradient(loss, self.network.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

                avg_train_loss += loss.numpy().mean()
                policy_accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(policy_targets, axis=1), tf.argmax(policy_preds, axis=1)), tf.float32))
                avg_train_accuracy += policy_accuracy.numpy()
                num_train_batches += 1

            avg_train_loss /= num_train_batches
            avg_train_accuracy /= num_train_batches

            # Evaluate the model on the validation data
            avg_val_loss = 0
            avg_val_accuracy = 0
            num_val_batches = 0
            for inputs, policy_targets, value_targets in val_dataloader:
                policy_preds, value_preds = self.network(inputs, training=False)
                value_loss = keras.losses.mean_squared_error(value_targets, value_preds)
                policy_loss = keras.losses.categorical_crossentropy(policy_targets, policy_preds)
                loss = value_loss + policy_loss

                avg_val_loss += loss.numpy().mean()
                policy_accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(policy_targets, axis=1), tf.argmax(policy_preds, axis=1)), tf.float32))
                avg_val_accuracy += policy_accuracy.numpy()
                num_val_batches += 1

            avg_val_loss /= num_val_batches
            avg_val_accuracy /= num_val_batches

            print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}')

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
                        layer_name[0:13] == 'value_flatten' or
                        layer_name[0:14] == 'policy_flatten' or
                        layer_name[0:10] == 'activation' or
                        layer_name[0:4] == 'relu' or
                        layer_name[0:8] == 'res_relu'):
                    continue
                layer_weights = weights_dict[layer_name]
                layer.set_weights([np.array(w) for w in layer_weights])

            print(f"Network weights loaded from Redis key '{key_name}'")

    def save_networks(self, key_name):
        self.save_network_weights(key_name)

    def load_networks(self, key_name):
        self.load_network_weights(key_name)

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

        print(f"Network weights saved to Redis key '{key_name}'")


def board_to_input(config, board_fen, chess_game_mcts):
    # Create an empty 8x8x17 tensor
    chess_game_mcts.board.set_fen(board_fen)
    input_tensor = np.zeros((config.board_size, config.board_size, config.num_channels))

    # Encode the current player in the first channel
    input_tensor[:, :, 0] = (chess_game_mcts.board.turn * 1.0)

    # Encode the piece positions in channels 2-13
    piece_map = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
    for square, piece in chess_game_mcts.board.piece_map().items():
        if piece.color == chess_game_mcts.WHITE:
            piece_idx = piece.piece_type - 1
        else:
            piece_idx = piece.piece_type - 1 + 6
        input_tensor[chess_game_mcts.square_rank(square), chess_game_mcts.square_file(square), piece_idx] = 1

    # Encode the fullmove number in channel 14
    input_tensor[:, :, 13] = chess_game_mcts.board.fullmove_number / 100.0

    # Encode the halfmove clock in channel 15
    input_tensor[:, :, 14] = chess_game_mcts.board.halfmove_clock / 100.0

    # Encode the remaining moves in channel 16
    remaining_moves = (2 * 50) - chess_game_mcts.board.fullmove_number
    input_tensor[:, :, 15] = remaining_moves / 100.0

    # Encode the remaining half-moves in channel 17
    remaining_halfmoves = 100 - chess_game_mcts.board.halfmove_clock
    input_tensor[:, :, 16] = remaining_halfmoves / 100.0

    return input_tensor


def get_legal_moves(board_fen, chess_game_mcts):
    """
    Return a list of all legal moves for the current player on the given board.
    """
    chess_game_mcts.board.set_fen(board_fen)
    legal_moves = list(chess_game_mcts.board.legal_moves)
    return [move.uci() for move in legal_moves]


# def move_to_index(move):
#     # Convert the move string to a chess.Move object
#     move_obj = chess.Move.from_uci(move)
#     # Convert the move object to an integer index
#     index = move_obj.from_square * 64 + move_obj.to_square
#     return index


class MCTSTree:
    # https://towardsdatascience.com/monte-carlo-tree-search-an-introduction-503d8c04e168
    def __init__(self, az):
        self.chess_game_mcts = ChessGame()
        node_list = self.create_nodes(az)
        self.root = node_list[0]
        self.network = az.network
        self.config = az.config

        self.unused_nodes = node_list[1:self.config.max_nodes]

    def create_nodes(self, az):
        all_nodes = list()
        for i in range(az.config.max_nodes):
            all_nodes.append(Node(board_fen=None, player_to_move=None, name='unused'))
        all_nodes[0].name = 'root'
        all_nodes[0].board_fen = self.chess_game_mcts.board.fen()
        all_nodes[0].player_to_move = 'white'
        return all_nodes

    def add_child_node(self, parent, board_fen, player_to_move, name):
        if len(self.unused_nodes) == 0:
            raise Exception('No more nodes available: expand config.max_nodes')
        node = self.unused_nodes.pop()
        node.parent = parent
        node.board_fen = board_fen
        node.player_to_move = player_to_move
        node.name = name
        node.parent.children.append(node)
        return node

    def delete_node(self, node):
        if node.parent is not None:
            if len(node.parent.children) > 0:
                node.parent.children.remove(node)
        node.reset_node()
        self.unused_nodes.append(node)

    def get_list_of_all_used_nodes(self):
        all_nodes = list()
        all_nodes.append(self.root)
        for node in all_nodes:
            for child in node.children:
                all_nodes.append(child)
        return all_nodes

    def get_policy_white(self, agent):
        # Get the policy from the root node
        epsilon = 1e-8
        policy = [child.Nvisit for child in self.root.children]
        policy_uci = [child.name for child in self.root.children]

        if any(math.isnan(pol) for pol in policy):
            policy = np.array([1 * self.root.children[i].game_over for i in range(len(self.root.children))])
        # Normalize the policy
        policy = np.array(policy) / (sum(policy) + epsilon)

        # Adjust the policy according to the temperature
        temp_adj_policy = np.power(policy, 1 / agent.temperature)
        temp_adj_policy /= np.sum(np.power(policy, 1 / agent.temperature)) + epsilon

        policy_array = policy_to_prob_array(policy, policy_uci,
                                            self.config.all_chess_moves)

        return policy, policy_uci, temp_adj_policy, policy_array

    def get_policy_black(self, agent):
        # Get the policy from the root node
        epsilon = 1e-8
        policy = [child.Nvisit for child in self.root.children]
        policy_uci = [child.name for child in self.root.children]

        if any(math.isnan(pol) for pol in policy):
            adjust_for_chess_game_mcts = 0
            policy = np.array([1 * self.root.children[i].game_over for i in range(len(self.root.children))])
        # Normalize the policy
        policy = np.array(policy) / (sum(policy) + epsilon)

        # Adjust the policy according to the temperature
        temp_adj_policy = np.power(policy, 1 / agent.temperature)
        temp_adj_policy /= np.sum(np.power(policy, 1 / agent.temperature))

        policy_array = policy_to_prob_array(policy, policy_uci,
                                            self.config.all_chess_moves)

        return policy, policy_uci, temp_adj_policy, policy_array

    def process_mcts(self, node, config, first_expand):
        epsilon = 1e-8
        policy = []
        if node.game_over:
            return policy
        # Select a node to expand
        if len(node.children) == 0:
            policy, first_expand = self.expand(node, first_expand)
            return policy

        # Evaluate the node
        max_uct = -float('inf')
        best_node = None

        adj = 1
        if node.player_to_move == 'black':
            adj = -1

        for child in node.children:
            uct = (adj * child.Qreward) + self.config.c_puct * child.prior_prob * math.sqrt(
                node.Nvisit + epsilon) / (1 + child.Nvisit)
            policy.append(child.prior_prob)

            if uct > max_uct:
                max_uct = uct
                best_node = child

        # Simulate a game from the best_node
        _ = self.process_mcts(best_node, config, first_expand)

        # Backpropagate the results of the simulation

        best_node.Qreward = (best_node.Qreward * best_node.Nvisit + best_node.prior_value) / (best_node.Nvisit + 1)
        best_node.Nvisit += 1

        node.Nvisit += 1
        del best_node, child
        return policy

    def expand(self, leaf_node, first_expand):
        # Get the policy and value from the neural network
        state = board_to_input(self.config, leaf_node.board_fen, self.chess_game_mcts)
        pi, v = self.network.predict(np.expand_dims(state, axis=0), verbose=0)
        del state
        leaf_node.prior_value = v[0][0]

        # Add Dirichlet noise to the prior probabilities
        if first_expand:
            alpha = self.config.dirichlet_alpha
            noise = np.random.dirichlet(alpha * np.ones(len(pi[0])))
            pi = (1 - self.config.eps) * pi[0] + self.config.eps * noise
            pi = np.array(pi) / sum(pi)
            first_expand = False
        else:
            pi = pi[0]
            pi = np.array(pi) / sum(pi)

        legal_moves = get_legal_moves(leaf_node.board_fen, self.chess_game_mcts)

        # Create list of legal policy probabilities corresponding to legal moves
        legal_probabilities = [pi[self.config.all_chess_moves.index(move)] for move in legal_moves]

        # Normalize the legal probabilities to sum to 1
        epsilon = 1e-8
        legal_probabilities /= (np.sum(legal_probabilities) + epsilon)

        for i, action in enumerate(legal_moves):
            new_board_fen = leaf_node.board_fen
            self.chess_game_mcts.board.set_fen(new_board_fen)
            self.chess_game_mcts.board.push_uci(action)
            if leaf_node.player_to_move == 'white':
                player_to_move = 'black'
            else:
                player_to_move = 'white'

            child = self.add_child_node(parent=leaf_node, board_fen=new_board_fen, name=action, player_to_move=player_to_move)
            self.chess_game_mcts.board.set_fen(child.board_fen)
            if self.chess_game_mcts.board.is_game_over(claim_draw=True):
                winner = self.chess_game_mcts.board.result()
                child.game_over = True
                if winner == '1-0':
                    child.prior_value = 1
                elif winner == '0-1':
                    child.prior_value = -1
                elif winner == '1/2-1/2':
                    if child.parent.player_to_move == 'black':
                        child.prior_value = -0.25
                    elif child.parent.player_to_move == 'white':
                        child.prior_value = 0.25

            child.prior_prob = legal_probabilities[i]

        del legal_moves, leaf_node, pi, v, new_board_fen, child
        return legal_probabilities, first_expand

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


def policy_to_prob_array(policy, legal_moves, all_moves_list):
    prob_array = np.zeros(len(all_moves_list))

    for i, move in enumerate(all_moves_list):
        if move in legal_moves:
            index = legal_moves.index(move)
            prob_array[i] = policy[index]

    return prob_array


class Node:
    def __init__(self, board_fen=None, player_to_move='white', name='root'):
        self.board_fen = board_fen
        self.Qreward = 0
        self.Nvisit = 0
        self.prior_prob = 0
        self.prior_value = 0
        self.children = []
        self.player_to_move = player_to_move
        self.parent = None
        self.game_over = False
        self.name = name

    def reset_node(self):
        self.name = 'unused'
        self.board_fen = None
        self.Qreward = 0
        self.Nvisit = 0
        self.prior_prob = 0
        self.prior_value = 0
        self.children = []
        if self.parent is not None:
            self.parent = None
        self.game_over = False

    def set_parent(self, parent):
        self.parent = weakref.ref(parent)

    def remove_from_all_nodes(self):
        for child in self.children:
            child.remove_from_all_nodes()
        Node.all_nodes.discard(self)
        del self

    def get_all_nodes(self):
        all_nodes = [weakref.ref(self)]
        for child in self.children:
            all_nodes += child.get_all_nodes()
        return all_nodes

    def count_nodes(self):
        # Recursively traverse the tree and increment the counter for each node
        count = 1  # Count the current node
        for child in self.children:
            count += child.count_nodes()[0]
        return count, len(Node.all_nodes)

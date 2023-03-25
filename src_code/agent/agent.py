import chess
import redis
import math
import copy
import random
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from src_code.agent.utils import draw_board
from src_code.agent.network import create_network


class AlphaZeroChess:
    def __init__(self, config, network=None):
        self.config = config
        self.board = chess.Board()
        # self.board = chess.Board(None)
        # # Place the black king at b8 (index 17)
        # self.board.set_piece_at(18, chess.Piece(chess.KING, chess.BLACK))
        #
        # # Place the white king at c6 (index 34)
        # self.board.set_piece_at(1, chess.Piece(chess.KING, chess.WHITE))
        #
        # # Place the white pawn at a2
        # self.board.set_piece_at(8, chess.Piece(chess.PAWN, chess.BLACK))

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
        self.board.reset()
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
        return policy_uci[action], policy, policy_array

    def game_over(self):
        return self.board.is_game_over(claim_draw=True)

    def get_result(self):
        # Check if the game is over
        if self.board.is_game_over(claim_draw=True):
            # Check if the game ended in checkmate
            if self.board.is_checkmate():
                # Return 1 if white wins, 0 if black wins
                return 1 if self.board.turn == chess.WHITE else -1
            # Otherwise, the game ended in stalemate or other draw
            elif self.board.is_stalemate() or self.board.is_fivefold_repetition() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves():
                return 0.5 if self.board.turn == chess.WHITE else -0.5
            else:
                return 0
        else:
            return 0

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


def board_to_input(config, board):
    # Create an empty 8x8x17 tensor
    input_tensor = np.zeros((config.board_size, config.board_size, config.num_channels))

    # Encode the current player in the first channel
    input_tensor[:, :, 0] = (board.turn * 1.0)

    # Encode the piece positions in channels 2-13
    piece_map = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
    for square, piece in board.piece_map().items():
        if piece.color == chess.WHITE:
            piece_idx = piece.piece_type - 1
        else:
            piece_idx = piece.piece_type - 1 + 6
        input_tensor[chess.square_rank(square), chess.square_file(square), piece_idx] = 1

    # Encode the fullmove number in channel 14
    input_tensor[:, :, 13] = board.fullmove_number / 100.0

    # Encode the halfmove clock in channel 15
    input_tensor[:, :, 14] = board.halfmove_clock / 100.0

    # Encode the remaining moves in channel 16
    remaining_moves = (2 * 50) - board.fullmove_number
    input_tensor[:, :, 15] = remaining_moves / 100.0

    # Encode the remaining half-moves in channel 17
    remaining_halfmoves = 100 - board.halfmove_clock
    input_tensor[:, :, 16] = remaining_halfmoves / 100.0

    return input_tensor


def make_move(board, uci_move, config):
    """
    Apply the given action to the given board and return the resulting board.
    """
    new_board = copy.deepcopy(board)
    new_board.push_uci(uci_move)
    new_state = board_to_input(config, new_board)
    return new_board, new_state


def get_legal_moves(board):
    """
    Return a list of all legal moves for the current player on the given board.
    """
    legal_moves = list(board.legal_moves)
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
        self.root = Node(az.board)
        self.network = az.network
        self.config = az.config

    def get_policy_white(self, agent):
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

    def get_policy_black(self, agent):
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
        temp_adj_policy /= np.sum(np.power(policy, 1 / agent.temperature))

        policy_array = policy_to_prob_array(policy, policy_uci,
                                            self.config.all_chess_moves)

        return policy, policy_uci, temp_adj_policy, policy_array

    def process_mcts(self, node, config, first_expand):
        epsilon = 1e-8
        policy = []
        if node.board.is_game_over(claim_draw=True):
            winner = node.board.result()
            node.game_over = True
            if winner == '1-0':
                node.prior_value = 1
            elif winner == '0-1':
                node.prior_value = -1
            elif winner == '1/2-1/2':
                if node.parent.player_to_move == 'black':
                    node.prior_value = -0.25
                elif node.parent.player_to_move == 'white':
                    node.prior_value = 0.25

            return policy
        # Select a node to expand
        if len(node.children) == 0:
            policy, first_expand = self.expand(node, first_expand)
            return policy

        # Evaluate the node
        max_uct = -float('inf')
        min_uct = float('inf')
        best_node = None

        for child in node.children:
            uct = child.Qreward + self.config.c_puct * child.prior_prob * math.sqrt(
                node.Nvisit + epsilon) / (1 + child.Nvisit)
            policy.append(child.prior_prob)
            if node.player_to_move == 'white':
                if uct > max_uct:
                    max_uct = uct
                    best_node = child
            elif node.player_to_move == 'black':
                if uct < min_uct:
                    min_uct = uct
                    best_node = child

        # Simulate a game from the best_node
        _ = self.process_mcts(best_node, config, first_expand)

        # Backpropagate the results of the simulation

        best_node.Qreward = (best_node.Qreward * best_node.Nvisit + best_node.prior_value) / (best_node.Nvisit + 1)
        best_node.Nvisit += 1

        node.Nvisit += 1
        return policy

    def expand(self, leaf_node, first_expand):
        # Get the policy and value from the neural network
        state = board_to_input(self.config, leaf_node.board)
        pi, v = self.network.predict(np.expand_dims(state, axis=0), verbose=0)
        leaf_node.prior_value = -1 * v[0][0]

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

        legal_moves = get_legal_moves(leaf_node.board)

        # Create list of legal policy probabilities corresponding to legal moves
        try:
            legal_probabilities = [pi[self.config.all_chess_moves.index(move)] for move in legal_moves]
        except:
            cwc = 0

        # Normalize the legal probabilities to sum to 1
        epsilon = 1e-8
        legal_probabilities /= (np.sum(legal_probabilities) + epsilon)

        for i, action in enumerate(legal_moves):
            new_board, new_state = make_move(copy.deepcopy(leaf_node.board), action, self.config)
            if leaf_node.player_to_move == 'white':
                player_to_move = 'black'
            else:
                player_to_move = 'white'
            child = Node(new_board, name=action, player_to_move=player_to_move)
            child.parent = leaf_node
            child.prior_prob = legal_probabilities[i]
            leaf_node.children.append(child)

        return legal_probabilities, first_expand

    def update_root(self, action):
        for child in self.root.children:
            if child.name == action:
                self.root = child
                self.root.parent = None
                self.root.name = 'root'
                break

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
    def __init__(self, board, player_to_move='white', name='root'):
        self.board = copy.deepcopy(board)
        self.Qreward = 0
        self.Nvisit = 0
        self.prior_prob = 0
        self.prior_value = 0
        self.children = []
        self.player_to_move = player_to_move
        self.parent = None
        self.game_over = False
        self.name = name

import chess
import redis
import math
import copy
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense
from src_code.agent.utils import draw_board


class AlphaZeroChess:
    def __init__(self, config):
        self.config = config
        self.board = chess.Board()
        self.num_channels = 17
        self.num_moves = 4096
        self.sim_counter = config.SimCounter()
        self.move_counter = config.MoveCounter()
        self.redis = redis.StrictRedis(host=config.redis_host, port=config.redis_port, db=config.redis_db)
        self.temperature = 1  # Starting value for temperature
        self.temperature_drop = 0.99  # How much to drop the temperature by each move
        self.min_temperature = 0.1  # Minimum temperature

        # Create the value network

        self.network = create_network(config)
        # Assign the optimizer to self.optimizer
        self.optimizer = config.optimizer
        self.state = board_to_input(config, self.board)

        # Initialize the MCTS tree
        self.tree = MCTSTree(self)

    def update_temperature(self):
        self.temperature = max(self.temperature * self.temperature_drop, self.min_temperature)

    def game_over(self):
        return self.board.is_game_over()

    def get_action(self, state):
        """Get the best action to take given the current state of the board."""
        action_probs, _ = self.tree.search()

        # Add Dirichlet noise to the action probabilities
        alpha = self.config.dirichlet_alpha
        noise = np.random.dirichlet(alpha * np.ones(self.config.action_space_size))
        action_probs = (1 - self.config.eps) * action_probs + self.config.eps * noise

        # Get the legal moves for the current board position
        legal_moves = get_legal_moves(self.board)

        # Filter out the illegal actions from the action probabilities
        legal_action_probs = np.zeros(self.config.action_space_size)
        for i in range(len(self.config.all_chess_moves)):
            if self.config.all_chess_moves[i] in legal_moves:
                legal_action_probs[i] = action_probs[i]
            else:
                legal_action_probs[i] = 0

        # Choose the action with the highest probability among the legal moves
        action = np.argmax(legal_action_probs)
        return action

    def get_policy(self, state):
        # Run the MCTS simulation and return the visit counts as the policy
        root = self.tree.root
        action_probs = np.zeros(self.config.action_space_size, dtype=np.float32)
        total_visits = 0
        for child in root.children:
            action_probs[self.config.all_chess_moves.index(child.name)] = child.visit_count
            total_visits += child.visit_count
        action_probs /= total_visits
        return action_probs

    def get_value(self, state):
        # Use the network to predict the value of the given state
        value = self.network.predict(np.expand_dims(state, axis=0))[1][0][0]
        return value

    def update_tree(self, state, action):
        """Update the MCTS tree with the latest state and action."""
        self.tree.update_root(state, action)

    def update_network(self, states, policy_targets, value_targets):
        """Update the neural network with the latest training data."""
        dataset = ChessDataset(states, policy_targets, value_targets)
        dataloader = tf.data.Dataset.from_generator(lambda: dataset, (tf.float32, tf.float32, tf.float32)).batch(self.config.batch_size)
        for inputs, policy_targets, value_targets in dataloader:
            with tf.GradientTape() as tape:
                value_preds, policy_preds = self.network(inputs)
                value_loss = keras.losses.mean_squared_error(value_targets, value_preds)
                policy_loss = keras.losses.categorical_crossentropy(policy_targets, policy_preds)
                loss = value_loss + policy_loss
            gradients = tape.gradient(loss, self.network.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        self.network.eval()

    def load_network_weights(self, key_name):
        # Connect to Redis and retrieve the serialized weights
        serialized_weights = self.redis.get(key_name)

        if serialized_weights is None:
            # Initialize the weights if no weights are found in Redis
            self.network = create_network(self.config)

            print(f"No weights found in Redis key '{key_name}'; network weights initialized")
        else:
            # Deserialize the weights from the byte string using NumPy
            weights_dict = np.loads(serialized_weights, allow_pickle=True)

            # Set the weights for each layer of the network
            for layer in self.network.layers:
                layer_name = layer.name
                layer_weights = weights_dict[layer_name]
                layer.set_weights([np.array(layer_weights[0]), np.array(layer_weights[1])])

            print(f"Network weights loaded from Redis key '{key_name}'")

    def save_network_weights(self, key_name):
        # Convert the weights to a dictionary
        weights_dict = {}
        for layer in self.network.layers:
            weights_dict[layer.name] = [layer.get_weights()[0].tolist(), layer.get_weights()[1].tolist()]

        # Serialize the dictionary to a byte string using NumPy
        serialized_weights = np.dumps(weights_dict)

        # Connect to Redis and save the weights using the specified key name

        self.redis.set(key_name, serialized_weights)

        print(f"Network weights saved to Redis key '{key_name}'")


class ChessDataset:
    def __init__(self, states, policy_targets, value_targets):
        self.states = states
        self.policy_targets = policy_targets
        self.value_targets = value_targets

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        state = self.states[index]
        policy_target = self.policy_targets[index]
        value_target = self.value_targets[index]
        return state, policy_target, value_target


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


def get_result(board):
    # Check if the game is over
    if board.is_game_over():
        # Check if the game ended in checkmate
        if board.is_checkmate():
            # Return 1 if white wins, 0 if black wins
            return 1 if board.turn == chess.WHITE else 0
        # Otherwise, the game ended in stalemate or other draw
        return 0.5
    # If the game is not over, return None
    return None


def make_move(board, action):
    """
    Apply the given action to the given board and return the resulting board.
    """
    new_board = copy.deepcopy(board)
    new_board.push_uci(action)
    return new_board


def get_legal_moves(board):
    """
    Return a list of all legal moves for the current player on the given board.
    """
    legal_moves = list(board.legal_moves)
    return [move.uci() for move in legal_moves]


def apply_move(config, board, action):
    """
    Apply the given action to the given board and return the resulting board state.
    """
    board.push_uci(action)
    return copy.deepcopy(board), board_to_input(config, board)


def move_to_index(move):
    # Convert the move string to a chess.Move object
    move_obj = chess.Move.from_uci(move)
    # Convert the move object to an integer index
    index = move_obj.from_square * 64 + move_obj.to_square
    return index


def residual_block(x, filters):
    y = Conv2D(filters, kernel_size=3, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters, kernel_size=3, padding='same')(y)
    y = BatchNormalization()(y)
    y = Add()([x, y])
    y = Activation('relu')(y)
    return y


# Define the neural network
def create_network(config):
    # Input layer
    inputs = Input(shape=(config.board_size, config.board_size, config.num_channels))

    # Residual blocks
    x = Conv2D(256, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    for i in range(4):
        x = residual_block(x, 256)

    # Value head
    v = Conv2D(1, kernel_size=1, padding='same')(x)
    v = BatchNormalization()(v)
    v = Activation('relu')(v)
    v = Flatten()(v)
    v = Dense(256, activation='relu')(v)
    v = Dense(1, activation='tanh', name='value')(v)

    # Policy head
    p = Conv2D(2, kernel_size=1, padding='same')(x)
    p = BatchNormalization()(p)
    p = Activation('relu')(p)
    p = Flatten()(p)
    p = Dense(config.action_space_size, activation='softmax', name='policy')(p)

    model = tf.keras.Model(inputs=inputs, outputs=[p, v])
    return model


class MCTSTree:
    def __init__(self, az):
        self.root = az.config.Node(az.state, az.board)
        self.network = az.network
        self.config = az.config

    def select(self, node):
        # Select the child node with the highest UCT value
        uct_values = []
        for child in node.children:
            q = child.total_value / child.visit_count if child.visit_count > 0 else 0
            p = child.prior_prob
            n = node.visit_count
            n_a = child.visit_count
            uct_value = q + self.config.c_puct * p * math.sqrt(n) / (1 + n_a)
            uct_values.append(uct_value)
        index = uct_values.index(max(uct_values))
        return node.children[index]

    def simulate(self, az):
        # Increment the simulation counter
        az.sim_counter.increment()
        # Assign the initial board state to the state variable
        state = board_to_input(self.config, az.board)
        # Simulate a game from the given state until the end using the policy network to select moves
        while not az.game_over():
            try:
                _, v = self.network.predict(np.expand_dims(state, axis=0))
            except Exception as e:
                print(f"Error occurred while predicting value: {e}")
                return None
            legal_moves = get_legal_moves(az.board)
            best_move = None
            best_value = -float('inf')
            for action in legal_moves:
                new_state = make_move(az.board, action)
                new_state = board_to_input(self.config, new_state)
                _, new_value = self.network.predict(np.expand_dims(new_state, axis=0))
                if new_value > best_value:
                    best_move = action
                    best_value = new_value
            az.board = make_move(az.board, best_move)
            v = best_value
        return get_result(az.board)

    def backpropogate(self, node, value):
        # Backpropagate the value of the end state up the tree
        node.visit_count += 1
        node.total_value += value
        if node.parent:
            self.backpropagate(node.parent, -value)

    def expand(self, node):
        # Generate all legal moves from the current state and create child nodes for each move
        pi, v = self.network.predict(np.expand_dims(node.state, axis=0))
        legal_moves = get_legal_moves(node.board)

        # Create list of legal policy probabilities corresponding to legal moves
        legal_probabilities = [pi[0][self.config.all_chess_moves.index(move)] for move in legal_moves]

        # Normalize the legal probabilities to sum to 1
        legal_probabilities /= np.sum(legal_probabilities)

        diff = 1.0 - np.sum(legal_probabilities)
        legal_probabilities[-1] += diff

        for i, action in enumerate(legal_moves):
            new_board, state = apply_move(self.config, copy.deepcopy(node.board), action)
            draw_board(new_board)

            child = self.config.Node(state, new_board, name=action)
            child.parent = node
            child.prior_prob = legal_probabilities[i]
            node.children.append(child)

    def search(self):
        # Run MCTS from the root node for a fixed number of iterations
        for i in range(self.config.num_iterations):
            node = self.root
            while node.children:
                node = self.select(node)
            if node.visit_count > 0:
                value = self.simulate(node.state)
                self.backpropagate(node, value)
            else:
                self.expand(node)
            if (i % 100) == 0:
                print(f'{i} tree searches complete.')

        # Get the action probabilities and value of the root node
        action_probs = np.zeros(self.config.action_space_size)
        for child in self.root.children:
            action_probs[self.config.all_chess_moves.index(child.name)] = child.visit_count / self.root.visit_count if self.root.visit_count > 0 else 0
        value = self.root.total_value / self.root.visit_count if self.root.visit_count > 0 else 0

        return action_probs, value

    def get_best_child(self, node, tau):
        legal_moves = get_legal_moves(node.board)
        legal_children = [child for child in node.children if child.name in legal_moves]
        visit_counts = np.array([child.visit_count for child in legal_children], dtype=np.float32)
        if not legal_children:
            # If there are no legal children, return None
            return None
        else:
            # Compute the visit count distribution with temperature scaling
            if np.sum(visit_counts) > 0:
                visit_dist = np.power(visit_counts, 1 / tau)
                visit_probs = visit_dist / np.sum(visit_dist)
            else:
                # If all visit counts are zero, assign a small positive probability to all child nodes
                visit_probs = np.ones(len(legal_children)) * 1e-6
                visit_probs /= np.sum(visit_probs)
            # Sample a child node from the visit count distribution
            index = np.random.choice(len(legal_children), p=visit_probs)
            return legal_children[index].name

    def get_children(self):
        """
        Return a list of all child nodes of the given node.
        """
        return [(child, action) for action, child in zip(get_legal_moves(self.state), self.children)]

    def update_root(self, state, action):
        # Find the child node corresponding to the given action
        for child in self.root.children:
            if child.name == action:
                self.root = child
                self.root.parent = None
                break
        else:
            # If the child node does not exist, create a new node and set it as the root
            uci_move = self.config.all_chess_moves[action]
            new_board, new_state = apply_move(self.config, copy.deepcopy(self.root.board), action)
            self.root = self.config.Node(new_state, new_board, name=action)


def generate_training_data(agent, config):
    # Initialize the lists to store the training data
    states = []
    policy_targets = []
    value_targets = []

    # Perform MCTS simulations to generate training data
    for i in range(config.num_sims):
        # Start a new simulation from the root node
        node = agent.tree.root
        sim_states = [board_to_input(config, agent.board)]

        # Perform the selection, expansion, simulation, and backpropagation steps of MCTS
        while node.children:
            node = agent.tree.select(node)
            uci_move = agent.tree.get_best_child(node, agent.temperature)
            if uci_move is None:
                break
            if uci_move not in get_legal_moves(agent.board):
                continue
            agent.board.push_uci(uci_move)
            sim_states.append(board_to_input(config, agent.board))
        agent.tree.expand(node)
        if (i % 100) == 0:
            print(f'{i} simulations complete.')

        # Get the value of the end state
        value = agent.tree.simulate(agent)

        # Backpropagate the value up the tree and collect the (state, policy, value) tuples
        for j in range(len(sim_states)):
            state = sim_states[j]
            policy = agent.tree.children[j].prior_prob
            states.append(state)
            policy_targets.append(policy)
            value_targets.append(value)

        # Reset the board and MCTS tree to the initial state
        agent.board.reset()
        agent.tree = MCTSTree(agent)

    return states, policy_targets, value_targets


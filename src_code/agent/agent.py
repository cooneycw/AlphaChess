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
    def __init__(self, config, redis_host='192.168.5.77', redis_port=6379):
        self.config = config
        self.board = chess.Board()
        self.num_channels = 17
        self.num_moves = 4096
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

        # Create the value network

        self.network = create_network(config)
        # Assign the optimizer to self.optimizer
        self.optimizer = config.optimizer
        self.state = board_to_input(config, self.board)

        # Initialize the MCTS tree
        self.tree = MCTSTree(self)

    def get_action(self, state):
        """Get the best action to take given the current state of the board."""
        action_probs, _ = self.tree.search()

        # Add Dirichlet noise to the action probabilities
        alpha = self.config.dirichlet_alpha
        noise = np.random.dirichlet(alpha * np.ones(self.config.action_space_size))
        action_probs = (1 - self.config.eps) * action_probs + self.config.eps * noise

        action = np.random.choice(len(action_probs), p=action_probs)
        return action

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


def game_over(board):
    return board.is_game_over()


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
    new_board.push(action)
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
        self.root = Node(az.state, az.board)
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

    def simulate(self, az, sim_counter):
        # Increment the simulation counter
        sim_counter.increment()
        # Assign the initial board state to the state variable
        state = az.board
        # Simulate a game from the given state until the end using the policy network to select moves
        while not game_over(state):
            _, v = self.network.predict(state)
            legal_moves = get_legal_moves(state)
            action_values = []
            for action in legal_moves:
                new_state = make_move(state, action)
                _, new_value = self.network.predict(new_state)
                action_values.append(new_value)
            action = legal_moves[np.argmax(action_values)]
        return get_result(state)

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

            child = Node(state, new_board, name=action)
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

        # Get the action probabilities and value of the root node
        action_probs = np.zeros(self.config.action_space_size)
        for child in self.root.children:
            action_probs[self.config.all_chess_moves.index(child.name)] = child.visit_count / self.root.visit_count if self.root.visit_count > 0 else 0
        value = self.root.total_value / self.root.visit_count if self.root.visit_count > 0 else 0

        return action_probs, value

    def get_best_action(self):
        # Select the best action based on the highest visit count of the child nodes
        values = [(child.visit_count, action) for action, child in self.get_children(self.root)]
        values.sort(reverse=True)
        return values[0][1]

    def get_children(self):
        """
        Return a list of all child nodes of the given node.
        """
        return [(child, action) for action, child in zip(get_legal_moves(self.state), self.children)]


class Node:
    def __init__(self, state, board, name='Game Start'):
        self.state = state
        self.board = copy.deepcopy(board)
        self.visit_count = 0
        self.total_value = 0
        self.prior_prob = 0
        self.children = []
        self.parent = None
        self.name = name


def generate_training_data(agent, config, sim_counter):
    # Initialize the lists to store the training data
    states = []
    policy_targets = []
    value_targets = []

    # Perform MCTS simulations to generate training data
    for i in range(config.num_simulations):
        # Start a new simulation from the root node
        node = agent.tree.root
        sim_states = [board_to_input(config, agent.board)]

        # Perform the selection, expansion, simulation, and backpropagation steps of MCTS
        while node.children:
            node = agent.tree.select(node)
            action = agent.tree.get_action(node)
            agent.board.push_uci(action)
            sim_states.append(board_to_input(config, agent.board))
        agent.tree.expand(node)

        # Get the value of the end state
        value = agent.tree.simulate(agent, sim_counter)

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

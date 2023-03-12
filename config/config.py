import chess
import copy
import tensorflow as tf


class Config:
    def __init__(self, verbosity):
        # Board and network settings
        self.board_size = 8
        self.num_channels = 17
        self.all_chess_moves = create_all_moves_list()
        self.self_play_games = 25000
        self.redis_host = '192.168.5.77'
        self.redis_port = 6379
        self.redis_db = 0
        # Training settings
        self.num_epochs = 10
        self.batch_size = 32
        self.maximum_moves = 100
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.num_iterations = 1600
        self.action_space_size = 4096
        self.dirichlet_alpha = 0.03  # Starting value for alpha
        self.eps = 0.25  # Starting value for eps
        self.num_sims = 800
        self.c_puct = 1.4
        self.alpha = 0.03
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)
        self.verbosity = verbosity
        self.SimCounter = SimulationCounter
        self.MoveCounter = MoveCounter
        self.GameCounter = GameCounter
        self.Node = Node


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


def create_all_moves_list():
    all_moves_list = []
    for square in chess.SQUARES:
        for target_square in chess.SQUARES:
            move = chess.Move(square, target_square)
            all_moves_list.append(move.uci())
    return all_moves_list


class SimulationCounter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def get_count(self):
        return self.count

    def reset(self):
        self.count = 0


class MoveCounter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def get_count(self):
        return self.count

    def reset(self):
        self.count = 0


class GameCounter:
    def __init__(self):
        self.count = 0

    def increment(self, n):
        self.count += n

    def get_count(self):
        return self.count

    def reset(self):
        self.count = 0

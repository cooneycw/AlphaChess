import chess
import tensorflow as tf



class Config:
    def __init__(self, verbosity):
        # Board and network settings
        self.board_size = 8
        self.num_channels = 17
        self.all_chess_moves = create_all_moves_list()
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


class MoveCounter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def get_count(self):
        return self.count

import tensorflow as tf


class Config:
    def __init__(self):
        # Board and network settings
        self.board_size = 8
        self.num_channels = 17
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

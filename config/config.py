import chess
import redis
import tensorflow as tf


class Config:
    def __init__(self, num_iterations, verbosity):
        # Board and network settings
        self.board_size = 8
        self.num_channels = 17
        self.all_chess_moves = create_all_moves_list()
        self.self_play_games = 10000
        self.redis_host = '192.168.5.77'
        self.redis_port = 6379
        self.redis_db = 0
        # Training settings
        self.num_epochs = 15
        self.validation_split = 0.1
        self.batch_size = 32
        self.maximum_moves = 180
        self.temperature = 1
        self.min_temperature = 0.01
        self.temperature_threshold = 150
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.num_iterations = num_iterations
        self.eval_num_iterations = 200
        self.num_evaluation_games = 100
        self.training_sample = 6000
        self.training_samples = 12
        self.early_stopping_epochs = 10
        self.reward_discount = 1.00
        self.action_space_size = 4096 + 176
        self.dirichlet_alpha = 0.3  # Starting value for alpha
        self.eps = 0.25  # Starting value for eps
        self.c_puct = 1.5
        self.eval_c_puct = 1.0
        # self.optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001)
        self.verbosity = verbosity
        self.SimCounter = SimulationCounter
        self.MoveCounter = MoveCounter
        self.game_counter = GameCounter()
        self.ChessDataset = ChessDataset


def create_all_moves_list():
    all_moves_list = []

    for square in chess.SQUARES:
        for target_square in chess.SQUARES:
            move = chess.Move(square, target_square)
            all_moves_list.append(move.uci())

    for promotion in [2, 3, 4, 5]:
        for square in chess.SQUARES:
            for target_square in chess.SQUARES:
                promotion_move = chess.Move(square, target_square, promotion=promotion)
                test_val = str(promotion_move)
                if len(test_val) > 4:
                    if (test_val[1] == '7' and test_val[3] == '8') or (test_val[1] == '2' and test_val[3] == '1'): # Promotion
                        if abs(ord(test_val[0]) - ord(test_val[2])) < 2:
                            all_moves_list.append(promotion_move.uci())

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

    def increment(self):
        self.count += 1

    def get_count(self):
        return self.count

    def reset(self):
        self.count = 0


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


def interpolate(start, end, t):
    return (1 - t) * start + t * end

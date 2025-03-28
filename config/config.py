import chess
import redis
import tensorflow as tf


class Config:
    def __init__(self, verbosity):
        # Board and network settings
        self.board_size = 8
        self.num_channels = 119
        self.all_chess_moves = create_all_moves_list()
        self.redis_host = '192.168.5.77'
        self.redis_port = 6379
        self.redis_db = 0
        # Training settings
        self.num_epochs = 1
        self.validation_split = 0.05
        self.batch_size = 48
        self.maximum_moves = 150
        self.temperature = 1
        self.min_temperature = 0.01
        self.temperature_threshold = self.maximum_moves
        self.initial_seed_games = 1000
        self.train_play_games = 1000
        self.eval_cycles = 800
        self.game_keys_limit = 250000
        self.num_iterations = 800
        self.eval_num_iterations = 800
        self.preplay_num_iterations = 300
        self.play_iterations = 40
        self.num_evaluation_games = 400
        self.reset_redis = False
        self.reset_network = True
        self.reset_initial = True
        self.initial_epochs = 16
        self.initial_early_stopping_epochs = 16
        self.training_sample = 4600 * 1
        self.training_samples = 1
        self.early_stopping_epochs = 1
        self.reward_discount = 1.00
        self.action_space_size = 4096 + 176
        self.dirichlet_alpha = 0.3  # Starting value for alpha
        self.eps = 0.25  # Starting value for eps
        self.c_puct = 1.5
        self.eval_c_puct = 1.2
        self.optimizer = None
        self.weight_decay = 0.00001
        self.max_gradient_norm = 1.0
        self.verbosity = verbosity
        self.SimCounter = SimulationCounter
        self.MoveCounter = MoveCounter
        self.game_counter = GameCounter()
        self.ChessDataset = ChessDataset

    def update_train_rate(self, learning_rate, opt_type):
        if opt_type == 'ada':
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif opt_type == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif opt_type == 'nadam':
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        elif opt_type == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif opt_type == 'adamax':
            self.optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        elif opt_type == 'adamw':
            self.optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=self.weight_decay)


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

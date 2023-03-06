class Config:
    def __init__(self):
        # Board and network settings
        self.board_size = 8
        self.num_channels = 256

        # Training settings
        self.num_epochs = 10
        self.batch_size = 32
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.num_iterations = 1600
        self.num_sims = 800
        self.c_puct = 1.4
        self.alpha = 0.03

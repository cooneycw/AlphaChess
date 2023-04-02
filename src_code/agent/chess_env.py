import chess


class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.WHITE = chess.WHITE
        self.BLACK = chess.BLACK
        self.square_rank = chess.square_rank
        self.square_file = chess.square_file

import chess
import ctypes
import chess.svg
import cairosvg
import io
import time
import inspect
import sys
import pickle
import datetime
import tkinter as tk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from pympler import asizeof


def draw_board(board, display=True, verbosity=False):
    if not verbosity:
        return
    board_svg = chess.svg.board(board)
    board_png = cairosvg.svg2png(bytestring=board_svg)
    if display == False:
        with open('board.png', 'wb') as f:
            f.write(board_png)
    else:
        root = tk.Tk()
        root.title('Chess Board')
        root.geometry('800x800')
        root.resizable(0, 0)
        board_image = ImageTk.PhotoImage(Image.open(io.BytesIO(board_png)))
        board_label = tk.Label(root, image=board_image)
        board_label.pack()
        root.mainloop()

    return


def generate_game_id():
    now = datetime.datetime.now()

    # Format the datetime as separate columns for date and time
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    game_id = date_str + '_' + time_str
    return game_id


def get_board_piece_count(board):
    num_white_pieces = len(board.pieces(chess.PAWN, chess.WHITE)) + \
                   len(board.pieces(chess.KNIGHT, chess.WHITE)) + \
                   len(board.pieces(chess.BISHOP, chess.WHITE)) + \
                   len(board.pieces(chess.ROOK, chess.WHITE)) + \
                   len(board.pieces(chess.QUEEN, chess.WHITE)) + \
                   len(board.pieces(chess.KING, chess.WHITE))

    # Count the number of black pieces
    num_black_pieces = len(board.pieces(chess.PAWN, chess.BLACK)) + \
                   len(board.pieces(chess.KNIGHT, chess.BLACK)) + \
                   len(board.pieces(chess.BISHOP, chess.BLACK)) + \
                   len(board.pieces(chess.ROOK, chess.BLACK)) + \
                   len(board.pieces(chess.QUEEN, chess.BLACK)) + \
                   len(board.pieces(chess.KING, chess.BLACK))
    return num_white_pieces, num_black_pieces


def save_training_data(agent, key_name, save_dict):
    # Connect to Redis and save the training data using the specified key name
    pickled_dict = pickle.dumps(save_dict)
    result = agent.redis.set(key_name, pickled_dict)
    if result is True:
        print(f"Training data saved to Redis key '{key_name}'  Value target:{save_dict['value_target']}")
    else:
        print(f"Error saving training data to Redis key '{key_name}'")


def load_training_data(agent, key_name, verbosity=True):
    # Connect to Redis and load the training data using the specified key name
    pickled_dict = agent.redis.get(key_name)
    save_dict = pickle.loads(pickled_dict)
    if verbosity:
        print(f"Training data loaded from Redis key '{key_name}'  Value target:{save_dict['value_target']}")
    return save_dict


def scan_redis_for_training_data(agent, match):
    key_list = []
    # Connect to Redis and scan for keys that start with 'training_data'
    keys = agent.redis.scan_iter(match=match+'*')
    for key in keys:
        key_list.append(key.decode('utf-8'))
    return key_list


def get_var_sizes(local_vars):
    for var, obj in local_vars:
        print(f'variable {var} size: {sys.getsizeof(obj)}')


def shrink_df(df, name, verbose):
    for i, column in enumerate(df.columns):
        if df[column].dtype == 'int64':
            df[column] = df[column].astype('int16')
            if verbose == True:
                print(f'Shrinking variable: {column}')
                print(f'Completed shrinking dataframe: {name}')
    return df


def print_variable_sizes_pympler(namespace):
    for name, value in namespace.items():
        size = asizeof(value)
        print(f"{name}: {size} bytes")


def find_variable_name(obj):
    frame = inspect.currentframe()
    for frame_info in inspect.getouterframes(frame):
        frame = frame_info[0]
        for name, value in frame.f_locals.items():
            if value is obj:
                return name
    return None


def get_size(referrers):
    for referrer in referrers:
        variable_name = find_variable_name(referrer)
        if variable_name:
            size = sys.getsizeof(referrer)
            variable_type = type(referrer)
            print(f"Object: {referrer}\nVariable name: {variable_name}\nType: {variable_type}\nSize: {size} bytes\n")


def malloc_trim():
    ctypes.CDLL('libc.so.6').malloc_trim(0)


def input_to_board(input_tensor):
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    def tensor_to_board(tensor_slice):
        board = chess.Board(empty=True)
        for rank in range(8):
            for file in range(8):
                for idx, piece_type in enumerate(piece_types):
                    if tensor_slice[rank, file, idx] == 1:
                        board.set_piece_at(chess.square(rank, file), chess.Piece(piece_type, chess.WHITE))
                    if tensor_slice[rank, file, idx + 6] == 1:
                        board.set_piece_at(chess.square(rank, file), chess.Piece(piece_type, chess.BLACK))
        return board

    board_list = []
    move_list = []
    for board_ind in range(8):
        board_slice = input_tensor[:, :, (board_ind * 14):(board_ind * 14) + 12]
        board = tensor_to_board(board_slice)
        board_list.append(board)

        if board_ind < 7:
            reps = input_tensor[0, 0, (board_ind * 14) + 12]
            opp_reps = input_tensor[0, 0, (board_ind * 14) + 13]
            move_list.append((reps, opp_reps))

    turn = bool(input_tensor[0, 0, 112])
    fullmove_number = int(input_tensor[0, 0, 113])

    white_kingside_castle = bool(input_tensor[0, 0, 114])
    white_queenside_castle = bool(input_tensor[0, 0, 115])
    black_kingside_castle = bool(input_tensor[0, 0, 116])
    black_queenside_castle = bool(input_tensor[0, 0, 117])

    halfmove_clock = int(input_tensor[0, 0, 118])

    current_play_summary = {
        "turn": turn,
        "fullmove_number": fullmove_number,
        "white_kingside_castle": white_kingside_castle,
        "white_queenside_castle": white_queenside_castle,
        "black_kingside_castle": black_kingside_castle,
        "black_queenside_castle": black_queenside_castle,
        "halfmove_clock": halfmove_clock,
    }

    return board_list, move_list, current_play_summary


def load_and_process_data(agent, verbosity, batch_size=5000):
    data_dict = {
        'states': [],
        'policy_targets': [],
        'value_targets': [],
    }
    keys = scan_redis_for_training_data(agent, 'azChess')
    # Load and process the selected keys
    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i:i+batch_size]
        batch_data = [load_training_data(agent, key, verbosity) for key in batch_keys]

        for data in batch_data:
            data_dict['states'].append(data['state'])
            data_dict['policy_targets'].append(data['policy_target'])
            data_dict['value_targets'].append(data['value_target'])

    data_dict['key_list'] = keys

    return data_dict

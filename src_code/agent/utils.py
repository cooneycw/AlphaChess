import chess
import chess.svg
import cairosvg
import io
import tkinter as tk
from PIL import Image, ImageTk


def draw_board(board, display=True):
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


def create_all_moves_dict(board):
    all_moves_dict = {}

    for square in chess.SQUARES:
        all_moves_dict[square] = []
        for target_square in chess.SQUARES:
            move = chess.Move(square, target_square)
            all_moves_dict[square].append(move.uci())
    return all_moves_dict

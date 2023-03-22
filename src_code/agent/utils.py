import chess
import chess.svg
import cairosvg
import io
import pickle
import datetime
import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image, ImageTk


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


def visualize_tree(tree):
    # Create a new graph
    graph = nx.DiGraph()

    # Add nodes to the graph
    add_node_to_graph(graph, tree.root)

    # Add edges to the graph
    add_edges_to_graph(graph, tree.root)

    # Draw the graph
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx(graph, pos)

    # Show the graph
    plt.show()


def add_node_to_graph(graph, node):
    # Add the node to the graph
    graph.add_node(node.name, label=node.name)


def add_edges_to_graph(graph, node):
    # Add edges from the node to its children
    for child in node.children:
        graph.add_edge(node.name, child.name)

        # Recursively add edges for the child's children
        add_edges_to_graph(graph, child)


def generate_game_id():
    now = datetime.datetime.now()
    game_id = now.strftime('%Y%m%d%H%M%S')
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


def load_training_data(agent, key_name):
    # Connect to Redis and load the training data using the specified key name
    pickled_dict = agent.redis.get(key_name)
    save_dict = pickle.loads(pickled_dict)
    print(f"Training data loaded from Redis key '{key_name}'  Value target:{save_dict['value_target']}")
    return save_dict


def scan_redis_for_training_data(agent, match):
    key_list = []
    # Connect to Redis and scan for keys that start with 'training_data'
    keys = agent.redis.scan_iter(match=match)
    for key in keys:
        key_list.append(key.decode('utf-8'))
    return key_list

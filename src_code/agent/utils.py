import chess
import chess.svg
import cairosvg
import io
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

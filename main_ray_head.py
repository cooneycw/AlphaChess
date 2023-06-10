import os
import logging
import ray
import socket
import copy
import random
import gc
import sys
import tensorflow as tf
from config.config import Config, interpolate
from src_code.utils.utils import get_non_local_ip
from src_code.agent.agent import AlphaZeroChess, board_to_input, create_network
from src_code.agent.self_play import play_games
from src_code.agent.train import train_model
from src_code.play.play import play_game
from src_code.evaluate.evaluate import run_evaluation
from src_code.evaluate.utils import scan_redis_for_networks, delete_redis_key
from src_code.agent.utils import draw_board, get_board_piece_count, generate_game_id, \
    save_training_data, load_training_data, scan_redis_for_training_data

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# conda activate alphatf
# ray start --head --num-cpus 10 --num-gpus 1 --dashboard-host 0.0.0.0
# ray start --address='192.168.5.132:6379'
USE_RAY = True

if USE_RAY:
    ray.init(logging_level=logging.INFO)


tf.get_logger().setLevel('ERROR')
physical_devices = tf.config.list_physical_devices('GPU')
if len(tf.config.list_physical_devices('GPU')) > 0:
    gpu_idx = 0  # Set the index of the GPU you want to use
    # Get the GPU device
    gpu_device = physical_devices[gpu_idx]
    # Set the GPU memory growth
    tf.config.experimental.set_memory_growth(gpu_device, True)
else:
    print('No GPUs available')

if __name__ == '__main__':

    input("Press Enter once remote workers have been initiated...")

    @ray.remote
    def worker_function():
        # Your worker logic here
        ray_worker_id = get_non_local_ip()
        message = f"Hello from worker: {ray_worker_id}"

        return message

    # Create a list to store the remote workers
    workers = []

    # Create 10,000 remote workers
    for _ in range(10000):
        worker = worker_function.remote()
        workers.append(worker)

    # Get the results from the workers
    results = ray.get(workers)

    # Print the message and IP address from each worker
    for result in results:
        message = result
        print(f"Message: {message}")

    ray.shutdown()

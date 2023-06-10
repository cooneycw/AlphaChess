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
from src_code.utils.utils import get_non_local_ip, total_cpu_workers
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


def initialize(in_config):
    network = create_network(in_config)
    outer_agent = AlphaZeroChess(in_config, network=network)
    outer_agent.save_networks('network_current')


@ray.remote
def main_ray(in_params):
    return main(in_params)


def main(in_params):
    if in_params['action'] == 'play':
        play_game()


if __name__ == '__main__':
    ray.init(logging_level=logging.INFO)

    outer_config = Config(verbosity=False)
    initialize(outer_config)

    # play seed games
    outer_agent = AlphaZeroChess(outer_config, network=None)
    outer_agent.load_networks('network_current')
    outer_agent.save_networks('network_latest')

    start_ind = 0
    num_workers = int(total_cpu_workers())
    while start_ind < outer_config.initial_seed_games:
        inds = list(range(start_ind, min(start_ind + num_workers, outer_config.initial_seed_games)))
        params_list = []

        for ind in inds:
            params_item = dict()
            params_item['action'] = 'play'
            params_item['config'] = copy.deepcopy(outer_config)
            params_item['game_id'] = ind
            params_list.append(params_item)

            results = [main_ray.remote(params_list[j]) for j in range(len(inds))]

            # Wait for all tasks to complete and get the results
            output = ray.get(results)

            # Print the output of each task
            for i, result in enumerate(output):
                print(f'Task {i} output: {result}')
                start_ind += 1

    ray.shutdown()

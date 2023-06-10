import os
import logging
import ray
import socket
import copy
import random
import gc
import sys
from config.config import Config
from src_code.utils.utils import total_cpu_workers, total_gpu_workers
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


def initialize(in_config):
    network = create_network(in_config)
    outer_agent = AlphaZeroChess(in_config, network=network)
    outer_agent.save_networks('network_current')


@ray.remote(num_gpus=0)
def main_ray_no_gpu(in_params):
    return main(in_params)


def main(in_params):
    if in_params['action'] == 'play':
        play_games(in_params)


if __name__ == '__main__':
    ray.init(logging_level=logging.INFO)
    verbosity = False

    outer_config = Config(verbosity=verbosity)

    # play seed games
    outer_agent = AlphaZeroChess(outer_config, network=None)
    outer_agent.redis.flushdb()
    initialize(outer_config)
    outer_agent.load_networks('network_current')

    start_ind = 0
    learning_rate = 0.2
    num_workers = int(total_cpu_workers())
    num_gpus = int(total_gpu_workers())
    print(f'Number of workers in ray cluster: {num_workers} gpus: {num_gpus}')
    while start_ind < outer_config.initial_seed_games:
        inds = list(range(start_ind, min(start_ind + num_workers, outer_config.initial_seed_games)))
        params_list = []

        for ind in inds:
            params_item = dict()
            params_item['action'] = 'play'
            params_item['verbosity'] = verbosity
            params_item['learning_rate'] = learning_rate
            params_item['network_name'] = 'network_current'
            params_item['game_id'] = ind
            params_list.append(params_item)

        results = [main_ray_no_gpu.remote(params_list[j]) for j in range(len(inds))]

        # Wait for all tasks to complete and get the results
        output = ray.get(results)

        # Print the output of each task
        for i, result in enumerate(output):
            print(f'Task {i} output: {result}')
            start_ind += 1

    ray.shutdown()

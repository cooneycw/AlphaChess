import logging
import ray
import copy
import chess
import gc
import datetime
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from config.config import Config, interpolate
from src_code.agent.agent import AlphaZeroChess, board_to_input, create_network
from src_code.agent.self_play import play_games
from src_code.agent.train import train_model
from src_code.evaluate.evaluate import evaluate_network
from src_code.agent.utils import draw_board, visualize_tree, get_board_piece_count, generate_game_id, save_training_data, load_training_data, scan_redis_for_training_data

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

USE_RAY = False
if USE_RAY:
    NUM_WORKERS = 1
    ray.init(num_cpus=NUM_WORKERS, num_gpus=0, ignore_reinit_error=True, logging_level=logging.DEBUG)

logging.getLogger('tensorflow').setLevel(logging.WARNING)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    gpu_idx = 0  # Set the index of the GPU you want to use
    tf.config.experimental.set_virtual_device_configuration(physical_devices[gpu_idx], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 1)])  # Set the memory limit (in bytes)
else:
    print('No GPUs available')


@ray.remote
def main_ray(in_params):
    return main(in_params)


def main(in_params):
    print(f'in_params: {in_params}')
    type = in_params['type']
    num_iterations = in_params['num_iterations']
    num_evals = in_params['num_evals']
    print(f'Running the main function with type: {type}')
    key_prefix = 'azChess_ThreadripperData_test'

    if type == 'create_training_data':
        game_id = generate_game_id()
        pass_dict = dict()
        pass_dict['game_id'] = game_id
        pass_dict['key_prefix'] = key_prefix
        pass_dict['num_iterations'] = num_iterations
        pass_dict['self_play_games'] = 1
        play_games(pass_dict)

        return f'Finished running the main function with type: {type} Game ID: {game_id}'

    elif type == 'train':
        train_model(key_prefix)
        return f'Finished running the main function with type: {type}'

    elif type == 'evaluate':
        pass_dict = dict()
        pass_dict['num_evals'] = num_evals
        pass_dict['num_iterations'] = num_iterations
        pass_dict['network_prefix'] = 'network_candidate_*'
        evaluate_network(pass_dict)
        return f'Finished running the main function with type: {type}'


def initialize(in_config):
    network = create_network(in_config)
    outer_agent = AlphaZeroChess(in_config, network=network)
    outer_agent.save_networks('network_current')


if __name__ == '__main__':
    type_list = ['initialize', 'create_training_data', 'train', 'evaluate']
    type_id = 3

    min_iterations = 2000
    outer_config = Config(num_iterations=min_iterations, verbosity=False)

    if type_list[type_id] == 'initialize':
        initialize(outer_config)

    if type_list[type_id] != 'initialize' and USE_RAY is False:
        params = dict()
        params['type'] = type_list[type_id]
        params['num_iterations'] = min_iterations
        params['num_evals'] = outer_config.num_evaluation_games
        params['self_play_games'] = 1
        outcome = main(params)
        print(f'Outcome: {outcome}')
    elif type_list[type_id] != 'initialize' and USE_RAY is True:

        max_num_iterations = 1600
        outer_config = Config(min_iterations, verbosity=False)

        start_ind = 0
        while start_ind < outer_config.self_play_games:
            inds = list(range(start_ind, min(start_ind + NUM_WORKERS, outer_config.self_play_games)))
            params_list = []

            for ind in inds:
                # Get the type ID for the current index
                params_item = dict()
                params_item['type'] = type_list[type_id]
                params_item['num_iterations'] = int(0.5 + interpolate(min_iterations, max_num_iterations, (min(ind, 5000)/5000)))
                params_item['num_evals'] = outer_config.num_evaluation_games
                params_item['self_play_games'] = 1
                params_list.append(params_item)

            results = [main_ray.remote(params_list[j]) for j in range(len(inds))]

            # Wait for all tasks to complete and get the results
            output = ray.get(results)

            # Print the output of each task
            for i, result in enumerate(output):
                print(f'Task {i} output: {result}')

            start_ind += NUM_WORKERS


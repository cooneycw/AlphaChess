import os
import logging
import ray
import socket
import copy
import random
import gc
import sys
import time
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


@ray.remote(num_gpus=1)
def main_ray_gpu(in_params):
    return main(in_params)


def main(in_params):
    if in_params['action'] == 'play':
        play_games(in_params)
    elif in_params['action'] == 'train':
        train_model(in_params)
    elif in_params['action'] == 'evaluate':
        run_evaluation(in_params)


if __name__ == '__main__':
    ray.init(logging_level=logging.INFO)
    verbosity = False

    outer_config = Config(verbosity=verbosity)

    # play seed games
    outer_agent = AlphaZeroChess(outer_config, network=None)
    outer_agent.redis.flushdb()
    initialize(outer_config)
    outer_agent.load_networks('network_current')

    print(f'Creating initial seed game data for training base.')
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

    agent_ind = 0
    while agent_ind < outer_config.eval_cycles:
        if agent_ind % 7 == 0 and agent_ind != 0:
            learning_rate = learning_rate * 0.1
        print(f'Executing train / game play iteration: {agent_ind} of {outer_config.eval_cycles}')
        pre_eval_ind = 0
        pre_eval_results = []
        while pre_eval_ind < outer_config.train_play_games:
            print(f'Executing post-train self play iteration: {pre_eval_ind} of {outer_config.train_play_games}')
            if pre_eval_ind == 0:
                network_name = 'network_current'
            else:
                network_name = 'network_current' + '_' + str(pre_eval_ind-1).zfill(5)
            network_name_out = 'network_current' + '_' + str(pre_eval_ind).zfill(5)
            train_params = dict()
            train_params['action'] = 'train'
            train_params['verbosity'] = verbosity
            train_params['network_name'] = network_name
            train_params['network_name_out'] = network_name_out
            train_params['learning_rate'] = learning_rate
            main(train_params)

            while True:
                # test number of ray workers / jobs currently running
                num_workers = int(total_cpu_workers())

                nodes = ray.nodes()
                # Calculate the total number of currently running worker jobs across all nodes.
                total_jobs = sum(len(node['Workers']) for node in nodes if 'Workers' in node)

                if num_workers > total_jobs:
                    params_item = dict()
                    params_item['action'] = 'play'
                    params_item['verbosity'] = verbosity
                    params_item['learning_rate'] = learning_rate
                    params_item['network_name'] = network_name_out
                    params_item['game_id'] = pre_eval_ind

                    print(f'Starting game {pre_eval_ind} of {outer_config.train_play_games}')
                    result_id = main_ray_no_gpu.remote(params_item)
                    pre_eval_results.append(result_id)
                    break
                else:
                    print(f'{total_jobs} of {num_workers} workers busy...waiting to start job')
                    time.time.sleep(5)

            pre_eval_ind += 1

        print(f'Training cycle completed.  Awaiting self-play.  Network evaluation follows.')
        results = [ray.get(result) for result in pre_eval_results]
        print(f'Self play completed.  Initiating evaluation process using {outer_config.num_evaluation_games} games.')

        eval_params = dict()
        eval_params['action'] = 'evaluate'
        eval_params['verbosity'] = verbosity
        eval_params['eval_game_id'] = None
        eval_params['random_val'] = None
        eval_params['network_current'] = 'network_current'
        eval_params['network_candidate'] = network_name_out

        challenger_wins = 0
        challenger_white_wins = 0
        challenger_black_wins = 0
        challenger_losses = 0
        challenger_draws = 0
        challenger_white_games = 0
        challenger_black_games = 0

        game_cnt = 0
        max_evals = outer_config.num_evaluation_games
        i = 0
        num_workers = int(total_cpu_workers())
        while i < max_evals:
            inds = [x for x in range(i, min(i + num_workers, max_evals))]
            eval_params_list = []
            for ind in inds:
                eval_params['eval_game_id'] = ind
                if ind % 2 == 0:
                    eval_params['random_val'] = 0.25
                else:
                    eval_params['random_val'] = 0.75
                eval_params_list.append(copy.deepcopy(eval_params))

            results = ray.get([main_ray_no_gpu.remote(eval_param) for eval_param in eval_params_list])

            for result in results:
                game_cnt += 1
                i += 1
                if result['player_to_go'] == 'candidate':
                    challenger_white_games += 1
                    challenger_white_wins += result['challenger_wins']
                else:
                    challenger_black_games += 1
                    challenger_black_wins += result['challenger_wins']
                challenger_wins += result['challenger_wins']
                challenger_losses += result['challenger_losses']
                challenger_draws += result['challenger_draws']

            if (game_cnt - challenger_draws) == 0:
                print(f'Games: {game_cnt} Wins: {challenger_wins} Losses: {challenger_losses} Draws: {challenger_draws}')
            else:
                print(f'Challenger wins: {challenger_wins} Losses: {challenger_losses} Draws: {challenger_draws}')
                print(f'Challenger white wins: {challenger_white_wins} of {challenger_white_games}')
                print(f'Challenger black wins: {challenger_black_wins} of {challenger_black_games}')
                print(f'Games: {game_cnt} Win/lose ratio: {0.1 * (int(0.5 + 1000 * challenger_wins / (game_cnt - challenger_draws)))}% ')

            gc_list = gc.get_objects()

        if (challenger_wins / (game_cnt - challenger_draws)) >= 0.55:
            print(f'Challenger won {0.1 * (int(0.5 + 1000 * challenger_wins / game_cnt))}% of the games')
            outer_agent.load_networks('network_current')
            outer_agent.save_networks('network_previous')
            outer_agent.load_networks(network_name_out)
            outer_agent.save_networks('network_current')
            outer_agent.save_networks('network_backup')
        else:
            print(f'Network: {network_name_out} was not adequate.  Deleting keys..')
            for k in range(0, outer_config.train_play_games):
                key = network_name + '_' + str(k).zfill(5)
                delete_redis_key(outer_agent, key)

        agent_ind += 1

    ray.shutdown()

import logging
import ray
import copy
import gc
import time
from config.config import Config
from src_code.utils.utils import total_cpu_workers, total_gpu_workers
from src_code.agent.agent import AlphaZeroChess, create_network
from src_code.agent.utils import load_and_process_data
from src_code.agent.self_play import play_games
from src_code.agent.train import train_model
from src_code.evaluate.evaluate import run_evaluation
from src_code.evaluate.utils import delete_redis_key


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# conda activate alphatf
# ray start --head --num-cpus 10 --num-gpus 1 --dashboard-host 0.0.0.0
# ray start --address='100.112.33.128:6379'


def initialize(in_config):
    network = create_network(in_config)
    init_agent = AlphaZeroChess(in_config,network=network)
    init_agent.save_networks('network_current')
    del init_agent


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
        result = run_evaluation(in_params)
        print(f'Result of evaluation {in_params["eval_game_id"]} is: {result}')
        return result


if __name__ == '__main__':
    ray.init(logging_level=logging.INFO)
    verbosity = False

    outer_config = Config(verbosity=verbosity)
    learning_rate = 0.0001
    opt_type = 'adamax'
    # play seed games
    outer_agent = AlphaZeroChess(outer_config, network=None)
    if outer_config.reset_redis is True:
        outer_agent.redis.flushdb()
    if outer_config.reset_network is True:
        initialize(outer_config)
    outer_agent.load_networks('network_current')

    if outer_config.reset_redis is True:
        print(f'Creating initial seed game data for training base.')
        start_ind = 0

        num_workers = int(total_cpu_workers())
        num_gpus = int(total_gpu_workers())
        print(f'Number of workers in ray cluster: {num_workers} gpus: {num_gpus}')
        running_tasks = []
        seed_results = []
        while start_ind < outer_config.initial_seed_games:
            while True:
                # test number of ray workers / jobs currently running
                num_workers = int(total_cpu_workers())

                # Check status of running tasks
                for task_info in running_tasks:
                    task_id = task_info
                    completed_tasks, _ = ray.wait([task_id], timeout=0)  # Check if task has finished
                    if len(completed_tasks) > 0:  # If list of completed tasks is non-empty
                        running_tasks.remove(task_info)  # Remove it from the list of running tasks

                if num_workers > len(running_tasks):
                    params_item = dict()
                    params_item['action'] = 'play'
                    params_item['pre_play'] = True
                    params_item['verbosity'] = verbosity
                    params_item['learning_rate'] = learning_rate
                    params_item['network_name'] = 'network_current'
                    params_item['game_id'] = start_ind
                    params_item['run_type'] = 'not_ray'

                    print(f'Starting game {start_ind} of {outer_config.initial_seed_games - 1}')
                    result_id = main_ray_no_gpu.remote(params_item)
                    running_tasks.append(result_id)
                    seed_results.append(result_id)
                    break
                else:
                    print(f'{len(running_tasks)} of {num_workers} workers busy...waiting to start job')
                    time.sleep(5)  # Note: it should be time.sleep(5) not time.time.sleep(5)

            start_ind += 1

        print(f'Seed cycle completed.  Awaiting seed self-play completion.  Regular training follows.')
        results = [ray.get(result) for result in seed_results]

    agent_ind = 0
    while agent_ind < outer_config.eval_cycles:
        if agent_ind % 42 == 0 and agent_ind != 0:
            learning_rate = learning_rate * 0.1
            opt_type = 'adam'
        print(f'Executing train / game play iteration: {agent_ind} of {outer_config.eval_cycles - 1}')
        pre_eval_ind = 0
        pre_eval_result_ids = []
        train_data = None
        running_tasks = []  # to store running tasks and their corresponding network_name_out values
        while pre_eval_ind < outer_config.train_play_games:
            print(f'Executing training step: {pre_eval_ind} of {outer_config.train_play_games - 1}')
            if pre_eval_ind == 0:
                network_name = 'network_current'
            else:
                network_name = 'network_current' + '_' + str(pre_eval_ind-1).zfill(5)
            network_name_out = 'network_current' + '_' + str(pre_eval_ind).zfill(5)

            if pre_eval_ind % 8 == 0 or train_data is None:
                print(f'retrieving redis game records...')
                train_data = load_and_process_data(outer_agent, verbosity)

            train_params = dict()
            train_params['action'] = 'train'
            train_params['verbosity'] = verbosity
            train_params['network_name'] = network_name
            train_params['network_name_out'] = network_name_out
            train_params['learning_rate'] = learning_rate
            train_params['opt_type'] = opt_type
            train_params['run_type'] = 'ray'
            train_params['train_data'] = train_data
            train_id = main_ray_gpu.remote(train_params)

            result = ray.get(train_id)

            while True:
                # test number of ray workers / jobs currently running
                num_workers = int(total_cpu_workers())

                # Check status of running tasks
                for task_info in running_tasks:
                    task_id, network_key_delete = task_info
                    completed_tasks, _ = ray.wait([task_id], timeout=0)  # Check if task has finished
                    if len(completed_tasks) > 0:  # If list of completed tasks is non-empty
                        running_tasks.remove(task_info)  # Remove it from the list of running tasks

                        if len(running_tasks) > 0:  # If there are still running tasks
                            # Delete the key from redis except for the very last job
                            if network_key_delete == 'network_current' + '_' + str(outer_config.train_play_games - 1).zfill(5):
                                pass
                            else:
                                delete_redis_key(outer_agent, network_key_delete)

                if num_workers > len(running_tasks):
                    params_item = dict()
                    params_item['action'] = 'play'
                    params_item['pre_play'] = False
                    params_item['verbosity'] = verbosity
                    params_item['learning_rate'] = learning_rate
                    params_item['network_name'] = network_name_out
                    params_item['game_id'] = pre_eval_ind
                    params_item['run_type'] = 'ray'

                    print(f'Starting game {pre_eval_ind} of {outer_config.train_play_games - 1}')
                    result_id = main_ray_no_gpu.remote(params_item)
                    pre_eval_result_ids.append(result_id)
                    running_tasks.append((result_id, network_name_out))
                    break
                else:
                    print(f'{len(running_tasks)} of {num_workers} workers busy...waiting to start job')
                    time.sleep(5)  # Note: it should be time.sleep(5) not time.time.sleep(5)

            pre_eval_ind += 1

        print(f'Training cycle completed.  Awaiting self-play completion.  Network evaluation follows.')
        # no need to wait for the training to complete if the final network model is ready for testing
        print(f'Self play completed.  Initiating evaluation process using {outer_config.num_evaluation_games} games.')

        eval_params = dict()
        eval_params['action'] = 'evaluate'
        eval_params['verbosity'] = verbosity
        eval_params['eval_game_id'] = None
        eval_params['rand_val'] = None
        eval_params['network_current'] = 'network_current'
        eval_params['network_candidate'] = network_name_out

        ind = 0
        eval_result_ids = []
        while ind < outer_config.num_evaluation_games:
            eval_params['eval_game_id'] = ind
            if ind % 2 == 0:
                eval_params['rand_val'] = 0.25
            else:
                eval_params['rand_val'] = 0.75

            while True:
                # test number of ray workers / jobs currently running
                num_workers = int(total_cpu_workers())

                # Check status of running tasks (should include pre-existing games since we did not await completion)
                for task_info in running_tasks:
                    task_id, _ = task_info
                    completed_tasks, _ = ray.wait([task_id], timeout=0)  # Check if task has finished
                    if len(completed_tasks) > 0:  # If list of completed tasks is non-empty
                        running_tasks.remove(task_info)  # Remove it from the list of running tasks

                if num_workers > len(running_tasks):
                    print(f'Starting evaluation game {ind} of {outer_config.num_evaluation_games - 1}')
                    result_id = main_ray_no_gpu.remote(eval_params)
                    running_tasks.append((result_id, " "))
                    eval_result_ids.append(result_id)
                    break
                else:
                    print(f'{len(running_tasks)} of {num_workers} workers busy...waiting to start job')
                    time.sleep(5)  # Note: it should be time.sleep(5) not time.time.sleep(5)

            ind += 1

        eval_results = [ray.get(eval_result_id) for eval_result_id in eval_result_ids]
        pre_eval_results = [ray.get(pre_eval_id) for pre_eval_id in pre_eval_result_ids]

        challenger_wins = 0
        challenger_white_wins = 0
        challenger_black_wins = 0
        challenger_losses = 0
        challenger_draws = 0
        challenger_white_games = 0
        challenger_black_games = 0

        game_cnt = 0

        for result in eval_results:
            game_cnt += 1

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

        if (challenger_wins / (game_cnt - challenger_draws)) >= 0.55:
            print(f'Challenger won {0.1 * (int(0.5 + 1000 * challenger_wins / game_cnt))}% of the games')
            outer_agent.load_networks('network_current')
            outer_agent.save_networks('network_previous')
            outer_agent.load_networks(network_name_out)
            outer_agent.save_networks('network_current')
            outer_agent.save_networks('network_backup')
        else:
            print(f'Network: {network_name_out} was not adequate.  Deleting keys..')
            delete_redis_key(outer_agent, network_name_out)

        agent_ind += 1

    ray.shutdown()

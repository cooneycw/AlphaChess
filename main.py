import logging
import ray
import copy
import gc
import tensorflow as tf
from config.config import Config, interpolate
from src_code.agent.agent import AlphaZeroChess, create_network
from src_code.agent.self_play import play_games
from src_code.agent.train import train_model
from play import play_game
from src_code.evaluate.evaluate import run_evaluation
from src_code.evaluate.utils import scan_redis_for_networks, delete_redis_key
from src_code.agent.utils import generate_game_id

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.get_logger().setLevel('ERROR')
USE_RAY = False
NUM_WORKERS = 10
NUM_GPUS = 0

if USE_RAY:
    ray.init(address=None, num_cpus=NUM_WORKERS, num_gpus=NUM_GPUS, logging_level=logging.INFO)

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


#@ray.remote(num_gpus=(NUM_GPUS/(2 + NUM_WORKERS)))
@ray.remote
def main_ray(in_params):
    return main(in_params)


def main(in_params):
    print(f'in_params: {in_params}')
    type = in_params['type']
    num_iterations = in_params['num_iterations']
    num_evals = in_params['num_evals']
    print(f'Running the main function with type: {type}')
    key_prefix = 'azChess_Threadripper00_prod'

    if type == 'create_training_data':
        game_id = generate_game_id()
        print(f'game_id:{game_id} spawned.')
        pass_dict = dict()
        pass_dict['game_id'] = game_id
        pass_dict['verbosity'] = False
        pass_dict['learning_rate'] = 0.2
        pass_dict['key_prefix'] = key_prefix
        pass_dict['num_iterations'] = num_iterations
        pass_dict['network_name'] = 'network_current_00038'
        pass_dict['self_play_games'] = num_evals
        pass_dict['run_type'] = 'not_ray'
        pass_dict['pre_play'] = False
        play_games(pass_dict)

        return f'Finished running the main function with type: {type} Game ID: {game_id}'

    elif type == 'train':
        pass_dict = dict()
        pass_dict['network_name'] = 'network_current_00003'
        pass_dict['network_name_out'] = 'network_test'
        pass_dict['learning_rate'] = 0.01
        pass_dict['verbosity'] = False
        pass_dict['opt_type'] = 'sgd'
        train_model(pass_dict)
        return f'Finished running the main function with type: {type}'

    elif type == 'play':
        play_game()
        return f'Finsihed running the main function with type: {type}'


def initialize(in_config):
    network = create_network(in_config)
    init_agent = AlphaZeroChess(in_config, network=network)
    init_agent.save_networks('network_current')
    del init_agent


if __name__ == '__main__':
    type_list = ['initialize', 'create_training_data', 'train', 'evaluate', 'play']
    type_id = 1

    min_iterations = 800
    outer_config = Config(verbosity=False)

    if type_list[type_id] == 'initialize':
        outer_agent = AlphaZeroChess(outer_config, network=None)
        initialize(outer_config)

    if type_list[type_id] != 'initialize' and USE_RAY is False:
        params = dict()
        params['type'] = type_list[type_id]
        params['num_iterations'] = min_iterations
        params['num_evals'] = outer_config.num_evaluation_games
        params['self_play_games'] = outer_config.train_play_games
        outcome = main(params)
        print(f'Outcome: {outcome}')

    elif type_list[type_id] == 'evaluate':
        assert USE_RAY is True, 'USE_RAY must be True'
        config = Config(num_iterations=800, verbosity=False)
        agent_admin = AlphaZeroChess(config)

        network_keys = scan_redis_for_networks(agent_admin, 'network_candidate_*')
        key = network_keys[0]
        print(f'Evaluating network {key} using {config.num_evaluation_games} games...')
        input_dict = dict()
        input_dict['eval_game_id'] = None
        input_dict['key'] = key

        challenger_wins = 0
        challenger_white_wins = 0
        challenger_black_wins = 0
        challenger_losses = 0
        challenger_draws = 0
        challenger_white_games = 0
        challenger_black_games = 0

        game_cnt = 0
        max_evals = config.num_evaluation_games
        i = 0
        while i < max_evals:
            inds = [x for x in range(i, min(i + NUM_WORKERS, max_evals))]
            print(f'Creating players corresponding to inds: {inds}')

            input_dict_list = []
            for ind in inds:
                input_dict['eval_game_id'] = ind
                if ind % 2 == 0:
                    input_dict['random_val'] = 0.25
                else:
                    input_dict['random_val'] = 0.75
                input_dict_list.append(copy.deepcopy(input_dict))

            results = ray.get([run_evaluation.remote(input_dict['eval_game_id'], input_dict['key'], input_dict['random_val']) for input_dict in input_dict_list])

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
            agent_admin.load_networks('network_current')
            agent_admin.save_networks('network_previous')
            agent_admin.load_networks(key)
            agent_admin.save_networks('network_current')
            agent_admin.save_networks('network_backup')
        else:
            print(f'Network: {key} was not adequate.  Deleting key..')
        delete_redis_key(agent_admin, key)

    elif type_list[type_id] != 'initialize' and USE_RAY is True:

        max_num_iterations = 800
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
                start_ind += 1

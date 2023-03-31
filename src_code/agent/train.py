import random
import itertools
import numpy as np
import datetime
from config.config import Config
from src_code.agent.agent import AlphaZeroChess
from src_code.agent.utils import scan_redis_for_training_data, load_training_data


def train_model(key_prefix, num_train_records=2000):
    config = Config(num_iterations=None, verbosity=False)
    agent = AlphaZeroChess(config)

    agent.load_networks('network_current')
    # Get a list of keys and shuffle them
    key_list = scan_redis_for_training_data(agent, key_prefix[0:7])
    shuffled_key_list = random.sample(list(key_list), len(list(key_list)))
    selected_keys = itertools.islice(shuffled_key_list, num_train_records)

    retrieved_data = []
    states = []
    policy_targets = []
    value_targets = []
    values = np.array(num_train_records * [0.0])

    # Load and process the selected keys
    for i, key in enumerate(selected_keys):
        retrieved_data.append(load_training_data(agent, key))
        states.append(retrieved_data[i]['state'])
        policy_targets.append(retrieved_data[i]['policy_target'])
        value_targets.append(retrieved_data[i]['value_target'])
        values[i] = retrieved_data[i]['value_target']

    win_abs_ratio = np.sum(np.abs(values)) / len(retrieved_data)
    win_ratio = np.sum(values) / len(retrieved_data)

    print(f'Loaded {len(retrieved_data)} training records')
    print(f'Sum of values: {np.sum(values)}  Win Absolute Ratio: {win_abs_ratio}  Win Ratio: {win_ratio}')

    agent.update_network(states, policy_targets, value_targets)
    now = datetime.datetime.now()

    # Format the datetime as separate columns for date and time
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    key_name = 'network_candidate_' + date_str + '_' + time_str
    agent.save_networks(key_name)

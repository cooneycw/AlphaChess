import random
import itertools
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from config.config import Config
from src_code.agent.agent import AlphaZeroChess
from src_code.agent.utils import scan_redis_for_training_data, load_training_data


def train_model(key_prefix, num_train_records=2000):
    config = Config(num_iterations=None, verbosity=False)
    agent = AlphaZeroChess(config)

    agent.load_networks('network_current')
    key_list = scan_redis_for_training_data(agent, key_prefix[0:7])
    # Get a list of keys and shuffle them

    retrieved_data = []
    states = []
    policy_targets = []
    value_targets = []
    values = np.array(len(list(key_list)) * [0.0])

    # Load and process the selected keys
    for i, key in enumerate(list(key_list)):
        retrieved_data.append(load_training_data(agent, key, config.verbosity))
        states.append(retrieved_data[i]['state'])
        policy_targets.append(retrieved_data[i]['policy_target'])
        value_targets.append(retrieved_data[i]['value_target'])
        values[i] = retrieved_data[i]['value_target']

    train_key_list, val_key_list, train_states, val_states, train_policy, val_policy, train_value, val_value = \
        split_data(config, key_list, states, policy_targets, value_targets)

    best_val = float('inf')
    last_n_val_losses = []
    for j in range(config.training_samples):
        num_train_samples = min(int(0.5 + (config.training_sample * (1-config.validation_split))), len(train_value))
        num_val_samples = min(int(0.5 + (config.training_sample * config.validation_split)), len(val_value))
        random_train_inds = random.sample(range(len(train_value)), num_train_samples)
        random_val_inds = random.sample(range(len(val_value)), num_val_samples)

        win_abs_ratio = np.sum(np.abs([train_value[i] for i in random_train_inds])) / len(random_train_inds)
        win_ratio = np.sum([train_value[i] for i in random_train_inds]) / len(random_train_inds)

        print(f'Sample: {j} of {config.training_samples}  Loaded {len(random_train_inds)} training records and {len(random_val_inds)} validation records')
        print(f'Sum of values: {np.sum([train_value[i] for i in random_train_inds])}  Win Absolute Ratio: {win_abs_ratio}%  Win Ratio: {win_ratio}%')

        validation_loss_tot, validation_loss_cnt = agent.update_network([train_states[i] for i in random_train_inds],
                                                                        [train_policy[i] for i in random_train_inds],
                                                                        [train_value[i] for i in random_train_inds],
                                                                        [val_states[j] for j in random_val_inds],
                                                                        [val_policy[j] for j in random_val_inds],
                                                                        [val_value[j] for j in random_val_inds])

        last_n_val_losses.append(validation_loss_tot/validation_loss_cnt)

        if len(last_n_val_losses) > config.early_stopping_epochs:
            last_n_val_losses.pop(0)  # Remove the oldest validation loss

        if len(last_n_val_losses) >= config.early_stopping_epochs:
            if last_n_val_losses[-1] < best_val:
                best_val = last_n_val_losses[-1]
                agent.save_networks('network_best_candidate')

        # Check if validation loss has not decreased for 'early_stopping_epochs' consecutive epochs
        if len(last_n_val_losses) == config.early_stopping_epochs and all(x <= last_n_val_losses[-1] for x
                                                                          in last_n_val_losses[:-1]):
            print(f"Early stopping triggered at training episode {j}")
            break

    now = datetime.datetime.now()

    # Format the datetime as separate columns for date and time
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    key_name = 'network_candidate_' + date_str + '_' + time_str
    agent.load_networks('network_best_candidate')
    agent.save_networks(key_name)

def split_data(config, key_list, states, policy_targets, value_targets):
    train_key_list, val_key_list, train_states, val_states, train_policy, val_policy, train_value, val_value = \
        train_test_split(key_list, states, policy_targets, value_targets, test_size=config.validation_split)
    return train_key_list, val_key_list, train_states, val_states, train_policy, val_policy, train_value, val_value

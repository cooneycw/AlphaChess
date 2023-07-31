import random
import redis
import numpy as np
import datetime
import gc
from sklearn.model_selection import train_test_split
from config.config import Config
from src_code.agent.agent import AlphaZeroChess
from src_code.agent.utils import scan_redis_for_training_data, load_training_data


def train_model(pass_dict):
    network_name = pass_dict['network_name']
    network_name_out = pass_dict['network_name_out']
    learning_rate = pass_dict['learning_rate']
    verbosity = pass_dict['verbosity']
    opt_type = pass_dict['opt_type']
    initial = pass_dict['initial']
    train_data = pass_dict['train_data']
    config = Config(verbosity=verbosity)
    config.update_train_rate(learning_rate, opt_type)
    agent = AlphaZeroChess(config)
    redis_conn = redis.Redis(host=config.redis_host, port=config.redis_port, db=config.redis_db)

    agent.load_networks(network_name)
    key_list = train_data['key_list']
    states = train_data['states']
    policy_targets = train_data['policy_targets']
    value_targets = train_data['value_targets']
    # Get a list of keys and shuffle them

    print(f'splitting records for test / train...')
    train_key_list, val_key_list, train_states, val_states, train_policy, val_policy, train_value, val_value = \
        split_data(config, key_list, states, policy_targets, value_targets)

    best_policy_val = float('inf')
    last_n_val_policy_losses = []
    best_set = False
    key_del_list = set()

    if initial:
        for k in range(config.initial_epochs):
            win_abs_ratio = np.sum(np.abs([train_value[ii] for ii in range(len(train_value))])) / len(train_value)
            win_ratio = np.sum([train_value[ii] for ii in range(len(train_value))]) / len(train_value)

            print(f'Epoch: {k}  Sum of values: {np.sum([train_value[ii] for ii in range(len(train_value))])}  Win Absolute Ratio: {int(100 * (0.005 + win_abs_ratio))}%  Win Ratio: {int(100 * (0.005 + win_ratio))}%')

            validation_loss_tot, validation_loss_cnt = agent.update_network([train_states[ii] for ii in range(len(train_states))],
                                                                            [train_policy[ii] for ii in range(len(train_policy))],
                                                                            [train_value[ii] for ii in range(len(train_value))],
                                                                            [val_states[jj] for jj in range(len(val_states))],
                                                                            [val_policy[jj] for jj in range(len(val_policy))],
                                                                            [val_value[jj] for jj in range(len(val_value))])

            last_n_val_policy_losses.append(validation_loss_tot / validation_loss_cnt)

            if len(last_n_val_policy_losses) > config.initial_early_stopping_epochs:
                last_n_val_policy_losses.pop(0)  # Remove the oldest validation loss

            if len(last_n_val_policy_losses) >= config.initial_early_stopping_epochs:
                if last_n_val_policy_losses[-1] < best_policy_val:
                    best_policy_val = last_n_val_policy_losses[-1]
                    agent.save_networks('network_best_candidate')
                    best_set = True

            # Check if validation loss has not decreased for 'early_stopping_epochs' consecutive epochs
            if len(last_n_val_policy_losses) == config.early_stopping_epochs and all(
                    x <= last_n_val_policy_losses[-1] for x
                    in last_n_val_policy_losses[:-1]):
                print(f"Early stopping triggered at training episode {k}")
                break
            gc.collect()

        if best_set is False:
            agent.save_networks('network_best_candidate')

        # Format the datetime as separate columns for date and time
        agent.load_networks('network_best_candidate')
        agent.save_networks(network_name_out)

        gc.collect()

    else:
        for j in range(config.training_samples):
            num_train_samples = min(int(0.5 + (config.training_sample * (1-config.validation_split))), len(train_value))
            num_val_samples = min(int(0.5 + (config.training_sample * config.validation_split)), len(val_value))
            random_train_inds = random.sample(range(len(train_value)), num_train_samples)
            random_val_inds = random.sample(range(len(val_value)), num_val_samples)

            win_abs_ratio = np.sum(np.abs([train_value[i] for i in random_train_inds])) / len(random_train_inds)
            win_ratio = np.sum([train_value[i] for i in random_train_inds]) / len(random_train_inds)

            print(f'Sample: {j+1} of {config.training_samples}  Sampled {len(random_train_inds)} training records and {len(random_val_inds)} validation records')
            print(f'Sum of values: {np.sum([train_value[i] for i in random_train_inds])}  Win Absolute Ratio: {int(100 * (0.005 + win_abs_ratio))}%  Win Ratio: {int(100 * (0.005 + win_ratio))}%')

            validation_loss_tot, validation_loss_cnt = agent.update_network([train_states[i] for i in random_train_inds],
                                                                            [train_policy[i] for i in random_train_inds],
                                                                            [train_value[i] for i in random_train_inds],
                                                                            [val_states[j] for j in random_val_inds],
                                                                            [val_policy[j] for j in random_val_inds],
                                                                            [val_value[j] for j in random_val_inds])

            gc.collect()
            last_n_val_policy_losses.append(validation_loss_tot / validation_loss_cnt)

            if len(last_n_val_policy_losses) > config.early_stopping_epochs:
                last_n_val_policy_losses.pop(0)  # Remove the oldest validation loss

            if len(last_n_val_policy_losses) >= config.early_stopping_epochs:
                if last_n_val_policy_losses[-1] < best_policy_val:
                    best_policy_val = last_n_val_policy_losses[-1]
                    agent.save_networks('network_best_candidate')
                    best_set = True

            # Remove sampled games from Redis
            for train_key in [train_key_list[i] for i in random_train_inds]:
                key_del_list.add(train_key)
            for val_key in [val_key_list[i] for i in random_val_inds]:
                key_del_list.add(val_key)

            # Check if validation loss has not decreased for 'early_stopping_epochs' consecutive epochs
            if len(last_n_val_policy_losses) == config.early_stopping_epochs and all(x <= last_n_val_policy_losses[-1] for x
                                                                              in last_n_val_policy_losses[:-1]):
                print(f"Early stopping triggered at training episode {j}")
                break
            gc.collect()

        if best_set is False:
            agent.save_networks('network_best_candidate')

        delete_keys(config, redis_conn, key_list, key_del_list)

        # Format the datetime as separate columns for date and time
        agent.load_networks('network_best_candidate')
        agent.save_networks(network_name_out)

        gc.collect()


def split_data(config, key_list, states, policy_targets, value_targets):
    train_key_list, val_key_list, train_states, val_states, train_policy, val_policy, train_value, val_value = \
        train_test_split(key_list, states, policy_targets, value_targets, test_size=config.validation_split)
    return train_key_list, val_key_list, train_states, val_states, train_policy, val_policy, train_value, val_value


def delete_keys(config, redis_conn, key_list, key_del_list):
    if len(key_list) > config.game_keys_limit:
        for q, keys in enumerate(list(key_del_list)):
            if q % 1 == 0:
                redis_conn.delete(keys)  # disable deletion of keys
        print(f'{len(key_del_list)} keys of {len(key_list)} keys deleted.')
    else:
        print(f'{len(key_list)} is below key limit of {config.game_keys_limit}.  No keys deleted.')

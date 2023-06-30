import redis
import datetime
import pickle
import numpy as np
from config.config import Config
from src_code.agent.agent import AlphaZeroChess
from src_code.agent.utils import input_to_board

# Connect to Redis
redis_client = redis.Redis(host='192.168.5.77', port=6379)
config = Config(num_iterations=800, verbosity=False)
# Get a list of all keys in the database
keys = redis_client.keys('*')

# Get a list of all keys that match the pattern
az_keys = [key.decode('utf-8') for key in redis_client.keys('az*')]
network_keys = [key.decode('utf-8') for key in redis_client.keys('network_candidate*')]
# key = network_keys[0]
# agent_admin = AlphaZeroChess(config)
# agent_admin.load_networks('network_current')
# agent_admin.save_networks('network_previous')
# agent_admin.load_networks(key)
# agent_admin.save_networks('network_current')
# agent_admin.save_networks('network_backup')
az_key_tuples = [tuple(key.split('_')) for key in az_keys]


for i, key in enumerate(az_keys):
    filter_values = az_key_tuples[i]
    value = pickle.loads(redis_client.get(key))
    board = binput_to_board(value['state'])
    move_index = np.argwhere(value['policy_target'] != 0).flatten()
    moves_and_values = [(config.all_chess_moves[index], value['policy_target'][index]) for index in move_index]
    moves_and_values.sort(key=lambda x: x[1], reverse=True)

    print(f'moves: {moves_and_values}')
    print(f'value: {value["value_target"]}')
    print(board)
    cwc = 0
# Filter the key tuples to only include tuples with a date greater than or equal to April 1, 2023
filtered_key_tuples = [key for key in az_key_tuples if datetime.datetime.strptime(key[3], '%Y-%m-%d') >= datetime.datetime(2023, 4, 1)]


print(f'Az keys: {len(az_keys)} network_keys: {len(network_keys)}')

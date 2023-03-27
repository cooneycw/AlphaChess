import redis

# Connect to Redis
redis_client = redis.Redis(host='192.168.5.77', port=6379)

# Get a list of all keys in the database
keys = redis_client.keys('*')

# Get a list of all keys that match the pattern
az_keys = redis_client.keys('az*')
network_keys = redis_client.keys('network*')

print(f'Az keys: {len(az_keys)} network_keys: {len(network_keys)}')

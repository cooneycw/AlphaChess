def scan_redis_for_networks(agent, match):
    key_list = []
    # Connect to Redis and scan for keys that start with 'training_data'
    keys = agent.redis.scan_iter(match=match+'*')
    for key in keys:
        key_list.append(key.decode('utf-8'))
    return key_list


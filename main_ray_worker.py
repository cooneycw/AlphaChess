import ray
import os
from src_code.utils.utils import get_non_local_ip

if __name__ == '__main__':
    worker = get_non_local_ip()
    if worker == '192.168.5.133':
        NUM_CPUS = 12
        NUM_GPUS = 0
    # Connect to the Ray cluster
    ray.init(address='192.168.5.132:6379', num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)  # Connects to the Ray head node automatically

    # Wait for tasks to be assigned and executed by the worker
    ray.worker.global_worker.run()



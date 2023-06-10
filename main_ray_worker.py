import ray
import time
from src_code.utils.utils import get_non_local_ip



if __name__ == '__main__':
    # Connect to the Ray cluster
    ray.init()  # Connects to the Ray head node automatically

    # Wait for tasks to be assigned and executed by the worker
    time.sleep(10000)



import ray
import os


os.environ["RAY_WORKER_ID"] = "120 Corsair 20tf"
# Connect to the Ray cluster
ray.init(address='192.168.5.132:6379')  # Connects to the Ray head node automatically

# Wait for tasks to be assigned and executed by the worker
ray.worker.global_worker.run()

ray.shutdown()

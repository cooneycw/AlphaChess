import ray

# Connect to the Ray cluster
ray.init(address='192.168.5.132:6379')  # Connects to the Ray head node automatically

# Wait for tasks to be assigned and executed by the worker
ray.worker.global_worker.run()

ray.shutdown()
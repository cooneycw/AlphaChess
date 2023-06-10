import netifaces
import ray
import tensorflow as tf


def get_non_local_ip():
    interfaces = netifaces.interfaces()
    for iface in interfaces:
        addresses = netifaces.ifaddresses(iface)
        if netifaces.AF_INET in addresses:
            for addr_info in addresses[netifaces.AF_INET]:
                ip_address = addr_info['addr']
                if ip_address != '127.0.0.1':
                    return ip_address
    return "Unknown"


def total_cpu_workers():
    resources = ray.cluster_resources()
    return resources['CPU']


def total_gpu_workers():
    resources = ray.cluster_resources()
    return resources['GPU']


def tensorflow_init():
    tf.get_logger().setLevel('ERROR')
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(tf.config.list_physical_devices('GPU')) > 0:
        gpu_idx = 0  # Set the index of the GPU you want to use
        # Get the GPU device
        gpu_device = physical_devices[gpu_idx]
        # Set the GPU memory growth
        tf.config.experimental.set_memory_growth(gpu_device, True)
    else:
        print('No GPUs available')


import netifaces
import ray


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

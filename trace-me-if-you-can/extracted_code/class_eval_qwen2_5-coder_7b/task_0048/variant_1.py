import socket
from functools import partial

class NetworkUtils:

    @staticmethod
    def validate_ip(ip_address, address_family):
        return socket.inet_pton(address_family, ip_address) is not None

    @staticmethod
    def is_valid_ipv4(ip_address):
        return NetworkUtils.validate_ip(ip_address, socket.AF_INET)

    @staticmethod
    def is_valid_ipv6(ip_address):
        return NetworkUtils.validate_ip(ip_address, socket.AF_INET6)

    @staticmethod
    def get_hostname(ip_address):
        try:
            return socket.gethostbyaddr(ip_address)[0]
        except socket.herror:
            return None

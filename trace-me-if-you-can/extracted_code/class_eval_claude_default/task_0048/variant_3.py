import socket
from contextlib import suppress


class IpUtil:

    @staticmethod
    def is_valid_ipv4(ip_address):
        with suppress(socket.error):
            socket.inet_pton(socket.AF_INET, ip_address)
            return True
        return False

    @staticmethod
    def is_valid_ipv6(ip_address):
        with suppress(socket.error):
            socket.inet_pton(socket.AF_INET6, ip_address)
            return True
        return False

    @staticmethod
    def get_hostname(ip_address):
        with suppress(socket.herror):
            return socket.gethostbyaddr(ip_address)[0]
        return None

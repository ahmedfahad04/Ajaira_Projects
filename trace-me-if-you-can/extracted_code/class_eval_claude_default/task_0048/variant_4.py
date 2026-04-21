import socket
from functools import wraps


def handle_socket_errors(default_return=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result if result is not None else True
            except (socket.error, socket.herror):
                return default_return
        return wrapper
    return decorator


class IpUtil:

    @staticmethod
    @handle_socket_errors(False)
    def is_valid_ipv4(ip_address):
        socket.inet_pton(socket.AF_INET, ip_address)

    @staticmethod
    @handle_socket_errors(False)
    def is_valid_ipv6(ip_address):
        socket.inet_pton(socket.AF_INET6, ip_address)

    @staticmethod
    @handle_socket_errors(None)
    def get_hostname(ip_address):
        return socket.gethostbyaddr(ip_address)[0]

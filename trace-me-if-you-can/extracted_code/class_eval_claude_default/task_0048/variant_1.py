import socket
from functools import partial


def _validate_ip_with_family(family, ip_address):
    try:
        socket.inet_pton(family, ip_address)
        return True
    except socket.error:
        return False


def _resolve_hostname(ip_address):
    try:
        return socket.gethostbyaddr(ip_address)[0]
    except socket.herror:
        return None


# Create specialized validators using partial application
is_valid_ipv4 = partial(_validate_ip_with_family, socket.AF_INET)
is_valid_ipv6 = partial(_validate_ip_with_family, socket.AF_INET6)
get_hostname = _resolve_hostname


class IpUtil:
    is_valid_ipv4 = staticmethod(is_valid_ipv4)
    is_valid_ipv6 = staticmethod(is_valid_ipv6)
    get_hostname = staticmethod(get_hostname)

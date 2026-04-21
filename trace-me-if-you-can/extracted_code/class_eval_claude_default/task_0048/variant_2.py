import socket


class IpUtil:
    
    _IP_FAMILIES = {
        'ipv4': socket.AF_INET,
        'ipv6': socket.AF_INET6
    }

    @classmethod
    def _validate_ip_by_type(cls, ip_type, ip_address):
        try:
            socket.inet_pton(cls._IP_FAMILIES[ip_type], ip_address)
            return True
        except socket.error:
            return False

    @staticmethod
    def is_valid_ipv4(ip_address):
        return IpUtil._validate_ip_by_type('ipv4', ip_address)

    @staticmethod
    def is_valid_ipv6(ip_address):
        return IpUtil._validate_ip_by_type('ipv6', ip_address)

    @staticmethod
    def get_hostname(ip_address):
        try:
            return socket.gethostbyaddr(ip_address)[0]
        except socket.herror:
            return None

import socket

class IPHandler:

    @staticmethod
    def is_valid(ip, family):
        with socket.socket(family, socket.SOCK_DGRAM) as s:
            try:
                s.connect((ip, 80))
                return True
            except socket.error:
                return False

    @staticmethod
    def is_valid_ipv4(ip):
        return IPHandler.is_valid(ip, socket.AF_INET)

    @staticmethod
    def is_valid_ipv6(ip):
        return IPHandler.is_valid(ip, socket.AF_INET6)

    @staticmethod
    def get_hostname(ip):
        try:
            return socket.gethostbyaddr(ip)[0]
        except socket.herror:
            return None

import socket

class IPAddressValidator:

    @staticmethod
    def check_valid_ipv4(ip):
        try:
            socket.inet_pton(socket.AF_INET, ip)
            return True
        except socket.error:
            return False

    @staticmethod
    def check_valid_ipv6(ip):
        try:
            socket.inet_pton(socket.AF_INET6, ip)
            return True
        except socket.error:
            return False

    @staticmethod
    def resolve_hostname(ip):
        try:
            return socket.gethostbyaddr(ip)[0]
        except socket.herror:
            return None

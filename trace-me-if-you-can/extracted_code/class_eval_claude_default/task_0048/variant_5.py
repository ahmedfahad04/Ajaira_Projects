import socket


class IpUtil:

    @staticmethod
    def _ip_validation_generator(ip_address, *families):
        for family in families:
            try:
                socket.inet_pton(family, ip_address)
                yield True
                return
            except socket.error:
                continue
        yield False

    @staticmethod
    def is_valid_ipv4(ip_address):
        return next(IpUtil._ip_validation_generator(ip_address, socket.AF_INET))

    @staticmethod
    def is_valid_ipv6(ip_address):
        return next(IpUtil._ip_validation_generator(ip_address, socket.AF_INET6))

    @staticmethod
    def get_hostname(ip_address):
        hostname_gen = (socket.gethostbyaddr(ip_address)[0] for _ in [None])
        try:
            return next(hostname_gen)
        except socket.herror:
            return None

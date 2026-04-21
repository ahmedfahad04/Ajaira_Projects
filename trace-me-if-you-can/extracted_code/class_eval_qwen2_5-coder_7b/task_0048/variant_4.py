import socket

class IPUtil:

    # Method to check if an IPv4 address is valid
    @staticmethod
    def is_valid_ipv4(ip_address):
        try:
            # Attempt to convert the IPv4 address to a packed binary format
            socket.inet_pton(socket.AF_INET, ip_address)
            return True
        except socket.error:
            # Return False if the conversion fails
            return False

    # Method to check if an IPv6 address is valid
    @staticmethod
    def is_valid_ipv6(ip_address):
        try:
            # Attempt to convert the IPv6 address to a packed binary format
            socket.inet_pton(socket.AF_INET6, ip_address)
            return True
        except socket.error:
            # Return False if the conversion fails
            return False

    # Method to get the hostname associated with an IP address
    @staticmethod
    def get_hostname(ip_address):
        try:
            # Attempt to get the hostname using the IP address
            hostname = socket.gethostbyaddr(ip_address)[0]
            return hostname
        except socket.herror:
            # Return None if the hostname cannot be found
            return None

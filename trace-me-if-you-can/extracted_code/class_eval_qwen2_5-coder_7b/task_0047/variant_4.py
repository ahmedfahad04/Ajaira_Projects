class IPHandler:
    def __init__(self, ip_address):
        self.ip_address = ip_address

    def is_ip_address_valid(self):
        octets = self.ip_address.split('.')
        if len(octets) != 4:
            return False
        for octet in octets:
            if not octet.isdigit() or not 0 <= int(octet) <= 255:
                return False
        return True

    def fetch_octets(self):
        if self.is_ip_address_valid():
            return self.ip_address.split('.')
        else:
            return []

    def to_binary_format(self):
        if self.is_ip_address_valid():
            binary_octets = ['{0:08b}'.format(int(octet)) for octet in self.fetch_octets()]
            return '.'.join(binary_octets)
        else:
            return ''

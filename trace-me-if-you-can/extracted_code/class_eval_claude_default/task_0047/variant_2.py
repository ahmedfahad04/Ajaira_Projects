import re

class IPAddress:
    IP_PATTERN = re.compile(r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$')
    
    def __init__(self, ip_address):
        self.ip_address = ip_address

    def is_valid(self):
        match = self.IP_PATTERN.match(self.ip_address)
        if not match:
            return False
        return all(0 <= int(group) <= 255 for group in match.groups())

    def get_octets(self):
        match = self.IP_PATTERN.match(self.ip_address)
        if match and all(0 <= int(group) <= 255 for group in match.groups()):
            return list(match.groups())
        return []

    def get_binary(self):
        octets = self.get_octets()
        if not octets:
            return ''
        return '.'.join(f'{int(octet):08b}' for octet in octets)

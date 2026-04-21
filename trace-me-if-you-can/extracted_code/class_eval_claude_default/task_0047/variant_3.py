class IPAddress:
    def __init__(self, ip_address):
        self.ip_address = ip_address

    def is_valid(self):
        def validate_octet(octet_str):
            return octet_str.isdigit() and 0 <= int(octet_str) <= 255

        parts = self.ip_address.split('.')
        return len(parts) == 4 and all(validate_octet(part) for part in parts)

    def get_octets(self):
        parts = self.ip_address.split('.')
        if len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts):
            return parts
        return []

    def get_binary(self):
        octets = self.get_octets()
        return '.'.join(bin(int(octet))[2:].zfill(8) for octet in octets) if octets else ''

class IPAddress:
    def __init__(self, ip_address):
        self.ip_address = ip_address

    def _parse_octets(self):
        try:
            octets = self.ip_address.split('.')
            if len(octets) != 4:
                raise ValueError("Invalid number of octets")
            
            parsed = []
            for octet in octets:
                if not octet.isdigit():
                    raise ValueError("Non-digit octet")
                value = int(octet)
                if value < 0 or value > 255:
                    raise ValueError("Octet out of range")
                parsed.append(octet)
            return parsed
        except ValueError:
            return None

    def is_valid(self):
        return self._parse_octets() is not None

    def get_octets(self):
        parsed = self._parse_octets()
        return parsed if parsed is not None else []

    def get_binary(self):
        octets = self.get_octets()
        if not octets:
            return ''
        binary_parts = [format(int(octet), 'b').rjust(8, '0') for octet in octets]
        return '.'.join(binary_parts)

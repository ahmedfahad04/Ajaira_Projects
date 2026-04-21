class IPAddress:
    def __init__(self, ip_address):
        self.ip_address = ip_address
        self._octets = None
        self._valid = None

    def is_valid(self):
        if self._valid is None:
            self._valid = self._validate_ip()
        return self._valid

    def _validate_ip(self):
        try:
            parts = self.ip_address.split('.')
            if len(parts) != 4:
                return False
            self._octets = []
            for part in parts:
                if not part.isdigit():
                    return False
                num = int(part)
                if not (0 <= num <= 255):
                    return False
                self._octets.append(part)
            return True
        except:
            return False

    def get_octets(self):
        return self._octets[:] if self.is_valid() else []

    def get_binary(self):
        if not self.is_valid():
            return ''
        return '.'.join(format(int(octet), '08b') for octet in self._octets)

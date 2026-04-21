class IPValidator:
    def __init__(self, address):
        self.address = address

    def validate(self):
        parts = self.address.split('.')
        if len(parts) != 4:
            return False
        for part in parts:
            if not part.isdigit() or not 0 <= int(part) <= 255:
                return False
        return True

    def extract_parts(self):
        return self.address.split('.') if self.validate() else []

    def to_binary(self):
        if self.validate():
            binary_parts = [format(int(part), '08b') for part in self.extract_parts()]
            return '.'.join(binary_parts)
        else:
            return ''

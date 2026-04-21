class IPv4:
    def __init__(self, address):
        self.address = address

    def check_validity(self):
        segments = self.address.split('.')
        if len(segments) != 4:
            return False
        for segment in segments:
            if not segment.isdigit() or not 0 <= int(segment) <= 255:
                return False
        return True

    def get_segments(self):
        if self.check_validity():
            return self.address.split('.')
        else:
            return []

    def convert_to_binary(self):
        if self.check_validity():
            binary_segments = ['{:08b}'.format(int(segment)) for segment in self.get_segments()]
            return '.'.join(binary_segments)
        else:
            return ''

class BinaryDataProcessor:
    def __init__(self, binary_string):
        self.binary_string = self._filter_binary_chars(binary_string)

    def _filter_binary_chars(self, data):
        return ''.join(char for char in data if char in '01')

    def calculate_binary_info(self):
        total_length = len(self.binary_string)
        ones_count = sum(1 for bit in self.binary_string if bit == '1')
        zeroes_count = total_length - ones_count

        return {
            'Zeroes': zeroes_count / total_length,
            'Ones': ones_count / total_length,
            'Bit length': total_length
        }

    def _bytes_from_binary(self):
        return bytearray(int(self.binary_string[i:i+8], 2) 
                        for i in range(0, len(self.binary_string), 8))

    def convert_to_ascii(self):
        return self._bytes_from_binary().decode('ascii')

    def convert_to_utf8(self):
        return self._bytes_from_binary().decode('utf-8')

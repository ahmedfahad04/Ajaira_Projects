import re
from functools import reduce

class BinaryDataProcessor:
    def __init__(self, binary_string):
        self.binary_string = re.sub(r'[^01]', '', binary_string)

    def calculate_binary_info(self):
        bit_counts = reduce(
            lambda acc, bit: (acc[0] + (bit == '0'), acc[1] + (bit == '1')),
            self.binary_string,
            (0, 0)
        )
        total_length = len(self.binary_string)
        
        return {
            'Zeroes': bit_counts[0] / total_length,
            'Ones': bit_counts[1] / total_length,
            'Bit length': total_length
        }

    def convert_to_ascii(self):
        chunks = [self.binary_string[i:i+8] for i in range(0, len(self.binary_string), 8)]
        bytes_data = bytes(int(chunk, 2) for chunk in chunks)
        return bytes_data.decode('ascii')

    def convert_to_utf8(self):
        chunks = [self.binary_string[i:i+8] for i in range(0, len(self.binary_string), 8)]
        bytes_data = bytes(int(chunk, 2) for chunk in chunks)
        return bytes_data.decode('utf-8')

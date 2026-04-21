from collections import Counter

class BinaryDataProcessor:
    def __init__(self, binary_string):
        self.original_string = binary_string
        self.binary_string = None
        self._process_input()

    def _process_input(self):
        valid_chars = []
        binary_set = set('01')
        
        for idx, char in enumerate(self.original_string):
            if char in binary_set:
                valid_chars.append(char)
        
        self.binary_string = ''.join(valid_chars)

    def calculate_binary_info(self):
        bit_counter = Counter(self.binary_string)
        total_bits = len(self.binary_string)
        
        zero_ratio = bit_counter['0'] / total_bits
        one_ratio = bit_counter['1'] / total_bits
        
        return {
            'Zeroes': zero_ratio,
            'Ones': one_ratio,
            'Bit length': total_bits
        }

    def convert_to_ascii(self):
        return self._convert_binary_to_string('ascii')

    def convert_to_utf8(self):
        return self._convert_binary_to_string('utf-8')

    def _convert_binary_to_string(self, target_encoding):
        byte_list = []
        string_length = len(self.binary_string)
        
        for start_idx in range(0, string_length, 8):
            end_idx = start_idx + 8
            binary_byte = self.binary_string[start_idx:end_idx]
            decimal_value = int(binary_byte, 2)
            byte_list.append(decimal_value)
        
        return bytearray(byte_list).decode(target_encoding)

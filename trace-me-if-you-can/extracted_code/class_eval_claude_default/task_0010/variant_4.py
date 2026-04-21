class BinaryDataProcessor:
    def __init__(self, binary_string):
        self.binary_string = binary_string
        self.clean_non_binary_chars()

    def clean_non_binary_chars(self):
        cleaned = ""
        for character in self.binary_string:
            if character == '0' or character == '1':
                cleaned += character
        self.binary_string = cleaned

    def calculate_binary_info(self):
        char_frequency = {}
        for bit in self.binary_string:
            char_frequency[bit] = char_frequency.get(bit, 0) + 1
        
        total_length = len(self.binary_string)
        zeroes_count = char_frequency.get('0', 0)
        ones_count = char_frequency.get('1', 0)

        return {
            'Zeroes': zeroes_count / total_length,
            'Ones': ones_count / total_length,
            'Bit length': total_length
        }

    def convert_to_ascii(self):
        return self._binary_to_text('ascii')

    def convert_to_utf8(self):
        return self._binary_to_text('utf-8')

    def _binary_to_text(self, encoding):
        byte_values = []
        current_pos = 0
        
        while current_pos + 8 <= len(self.binary_string):
            byte_chunk = self.binary_string[current_pos:current_pos + 8]
            byte_values.append(int(byte_chunk, 2))
            current_pos += 8
            
        return bytearray(byte_values).decode(encoding)

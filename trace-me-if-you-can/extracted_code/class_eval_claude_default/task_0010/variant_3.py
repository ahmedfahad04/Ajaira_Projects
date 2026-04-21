class BinaryDataProcessor:
    VALID_BITS = {'0', '1'}
    
    def __init__(self, binary_string):
        self.binary_string = binary_string
        self._sanitize_input()

    def _sanitize_input(self):
        sanitized = []
        for char in self.binary_string:
            if char in self.VALID_BITS:
                sanitized.append(char)
        self.binary_string = ''.join(sanitized)

    def calculate_binary_info(self):
        stats = {'0': 0, '1': 0}
        for bit in self.binary_string:
            stats[bit] += 1
        
        total_length = len(self.binary_string)
        return {
            'Zeroes': stats['0'] / total_length,
            'Ones': stats['1'] / total_length,
            'Bit length': total_length
        }

    def _decode_with_encoding(self, encoding):
        result = bytearray()
        bit_position = 0
        while bit_position < len(self.binary_string):
            byte_slice = self.binary_string[bit_position:bit_position + 8]
            result.append(int(byte_slice, 2))
            bit_position += 8
        return result.decode(encoding)

    def convert_to_ascii(self):
        return self._decode_with_encoding('ascii')

    def convert_to_utf8(self):
        return self._decode_with_encoding('utf-8')

class BinaryStringHandler:
    def __init__(self, binary_str):
        self.binary_str = binary_str
        self.strip_invalid_chars()

    def strip_invalid_chars(self):
        self.binary_str = ''.join(filter(lambda x: x in '01', self.binary_str))

    def compute_binary_stats(self):
        zero_count = self.binary_str.count('0')
        one_count = self.binary_str.count('1')
        total_length = len(self.binary_str)

        zero_ratio = zero_count / total_length
        one_ratio = one_count / total_length

        return {
            'Zeros': zero_ratio,
            'Ones': one_ratio,
            'Bit length': total_length
        }

    def transform_to_ascii(self):
        ascii_str = ''
        for i in range(0, len(self.binary_str), 8):
            byte = self.binary_str[i:i+8]
            decimal_value = int(byte, 2)
            ascii_str += chr(decimal_value)

        return ascii_str

    def transform_to_utf8(self):
        utf8_str = ''
        for i in range(0, len(self.binary_str), 8):
            byte = self.binary_str[i:i+8]
            decimal_value = int(byte, 2)
            utf8_str += chr(decimal_value)

        return utf8_str

class BitStringParser:
    def __init__(self, bit_str):
        self.bit_str = bit_str
        self.purge_non_binary()

    def purge_non_binary(self):
        self.bit_str = ''.join(filter(lambda x: x in '01', self.bit_str))

    def binary_statistics(self):
        zero_count = self.bit_str.count('0')
        one_count = self.bit_str.count('1')
        total_length = len(self.bit_str)

        zero_percentage = zero_count / total_length
        one_percentage = one_count / total_length

        return {
            'Zeros': zero_percentage,
            'Ones': one_percentage,
            'Bit length': total_length
        }

    def binary_to_ascii_text(self):
        ascii_string = ''
        for i in range(0, len(self.bit_str), 8):
            byte = self.bit_str[i:i+8]
            decimal_value = int(byte, 2)
            ascii_string += chr(decimal_value)

        return ascii_string

    def binary_to_utf8_text(self):
        utf8_string = ''
        for i in range(0, len(self.bit_str), 8):
            byte = self.bit_str[i:i+8]
            decimal_value = int(byte, 2)
            utf8_string += chr(decimal_value)

        return utf8_string

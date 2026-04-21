class BitDataManipulator:
    def __init__(self, bit_data):
        self.bit_data = bit_data
        self.clean_non_bit_chars()

    def clean_non_bit_chars(self):
        self.bit_data = ''.join(filter(lambda x: x in '01', self.bit_data))

    def calculate_bit_info(self):
        zero_freq = self.bit_data.count('0')
        one_freq = self.bit_data.count('1')
        total_length = len(self.bit_data)

        zero_ratio = zero_freq / total_length
        one_ratio = one_freq / total_length

        return {
            'Zeros': zero_ratio,
            'Ones': one_ratio,
            'Bit length': total_length
        }

    def convert_to_ascii_string(self):
        ascii_result = ''
        for i in range(0, len(self.bit_data), 8):
            byte = self.bit_data[i:i+8]
            decimal = int(byte, 2)
            ascii_result += chr(decimal)

        return ascii_result

    def convert_to_utf8_string(self):
        utf8_result = ''
        for i in range(0, len(self.bit_data), 8):
            byte = self.bit_data[i:i+8]
            decimal = int(byte, 2)
            utf8_result += chr(decimal)

        return utf8_result

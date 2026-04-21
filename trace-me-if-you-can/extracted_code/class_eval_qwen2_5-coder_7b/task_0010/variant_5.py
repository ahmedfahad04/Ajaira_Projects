class BinaryInfoExtractor:
    def __init__(self, binary_data):
        self.binary_data = binary_data
        self.remove_invalid_chars()

    def remove_invalid_chars(self):
        self.binary_data = ''.join(filter(lambda x: x in '01', self.binary_data))

    def get_binary_details(self):
        zero_count = self.binary_data.count('0')
        one_count = self.binary_data.count('1')
        total_length = len(self.binary_data)

        zero_percentage = zero_count / total_length
        one_percentage = one_count / total_length

        return {
            'Zeros': zero_percentage,
            'Ones': one_percentage,
            'Bit length': total_length
        }

    def binary_to_ascii_output(self):
        ascii_output = ''
        for i in range(0, len(self.binary_data), 8):
            byte = self.binary_data[i:i+8]
            decimal_value = int(byte, 2)
            ascii_output += chr(decimal_value)

        return ascii_output

    def binary_to_utf8_output(self):
        utf8_output = ''
        for i in range(0, len(self.binary_data), 8):
            byte = self.binary_data[i:i+8]
            decimal_value = int(byte, 2)
            utf8_output += chr(decimal_value)

        return utf8_output

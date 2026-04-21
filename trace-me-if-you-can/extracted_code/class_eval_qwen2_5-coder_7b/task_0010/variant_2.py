class BinaryProcessor:
    def __init__(self, binary):
        self.binary = binary
        self.remove_non_binary()

    def remove_non_binary(self):
        self.binary = ''.join(filter(lambda x: x in '01', self.binary))

    def analyze_binary(self):
        zero_freq = self.binary.count('0')
        one_freq = self.binary.count('1')
        total_length = len(self.binary)

        zero_ratio = zero_freq / total_length
        one_ratio = one_freq / total_length

        return {
            'Zeroes': zero_ratio,
            'Ones': one_ratio,
            'Bit length': total_length
        }

    def binary_to_ascii(self):
        ascii_text = ''
        for i in range(0, len(self.binary), 8):
            byte = self.binary[i:i+8]
            decimal = int(byte, 2)
            ascii_text += chr(decimal)

        return ascii_text

    def binary_to_utf8(self):
        utf8_text = ''
        for i in range(0, len(self.binary), 8):
            byte = self.binary[i:i+8]
            decimal = int(byte, 2)
            utf8_text += chr(decimal)

        return utf8_text

class NumberConverter:
    @staticmethod
    def dec_to_bin(num):
        return bin(num)[2:]

    @staticmethod
    def bin_to_dec(num):
        return int(num, 2)

    @staticmethod
    def dec_to_oct(num):
        return oct(num)[2:]

    @staticmethod
    def oct_to_dec(num):
        return int(num, 8)

    @staticmethod
    def dec_to_hex(num):
        return hex(num)[2:]

    @staticmethod
    def hex_to_dec(num):
        return int(num, 16)

class NumberConverter:
    @staticmethod
    def _convert(num, func):
        return func(num)

    @staticmethod
    def decimal_to_binary(num):
        return NumberConverter._convert(num, lambda x: bin(x)[2:])

    @staticmethod
    def binary_to_decimal(num):
        return NumberConverter._convert(num, lambda x: int(x, 2))

    @staticmethod
    def decimal_to_octal(num):
        return NumberConverter._convert(num, lambda x: oct(x)[2:])

    @staticmethod
    def octal_to_decimal(num):
        return NumberConverter._convert(num, lambda x: int(x, 8))

    @staticmethod
    def decimal_to_hex(num):
        return NumberConverter._convert(num, lambda x: hex(x)[2:])

    @staticmethod
    def hex_to_decimal(num):
        return NumberConverter._convert(num, lambda x: int(x, 16))

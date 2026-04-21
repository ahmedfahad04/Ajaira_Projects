class NumberConverter:
    @staticmethod
    def decimal_to_binary(decimal_num):
        if decimal_num == 0:
            return '0'
        result = ''
        while decimal_num > 0:
            result = str(decimal_num % 2) + result
            decimal_num //= 2
        return result

    @staticmethod
    def binary_to_decimal(binary_num):
        decimal_num = 0
        power = 0
        for digit in reversed(binary_num):
            decimal_num += int(digit) * (2 ** power)
            power += 1
        return decimal_num

    @staticmethod
    def decimal_to_octal(decimal_num):
        if decimal_num == 0:
            return '0'
        result = ''
        while decimal_num > 0:
            result = str(decimal_num % 8) + result
            decimal_num //= 8
        return result

    @staticmethod
    def octal_to_decimal(octal_num):
        decimal_num = 0
        power = 0
        for digit in reversed(octal_num):
            decimal_num += int(digit) * (8 ** power)
            power += 1
        return decimal_num

    @staticmethod
    def decimal_to_hex(decimal_num):
        if decimal_num == 0:
            return '0'
        hex_chars = '0123456789abcdef'
        result = ''
        while decimal_num > 0:
            result = hex_chars[decimal_num % 16] + result
            decimal_num //= 16
        return result

    @staticmethod
    def hex_to_decimal(hex_num):
        hex_values = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15}
        decimal_num = 0
        power = 0
        for digit in reversed(hex_num.lower()):
            decimal_num += hex_values[digit] * (16 ** power)
            power += 1
        return decimal_num

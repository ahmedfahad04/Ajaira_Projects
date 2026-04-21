class NumberConverter:
    _CONVERSION_MAP = {
        'decimal_to_binary': lambda x: bin(x)[2:],
        'binary_to_decimal': lambda x: int(x, 2),
        'decimal_to_octal': lambda x: oct(x)[2:],
        'octal_to_decimal': lambda x: int(x, 8),
        'decimal_to_hex': lambda x: hex(x)[2:],
        'hex_to_decimal': lambda x: int(x, 16)
    }

    @staticmethod
    def convert(num, from_base, to_base):
        if f'{from_base}_to_{to_base}' not in NumberConverter._CONVERSION_MAP:
            raise ValueError("Invalid conversion type")
        return NumberConverter._CONVERSION_MAP[f'{from_base}_to_{to_base}'](num)

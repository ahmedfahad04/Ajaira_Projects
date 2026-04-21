from enum import Enum

class ConversionType(Enum):
    DEC_TO_BIN = 'decimal_to_binary'
    BIN_TO_DEC = 'binary_to_decimal'
    DEC_TO_OCT = 'decimal_to_octal'
    OCT_TO_DEC = 'octal_to_decimal'
    DEC_TO_HEX = 'decimal_to_hex'
    HEX_TO_DEC = 'hex_to_decimal'

class NumberConverter:
    _CONVERSION_MAP = {
        ConversionType.DEC_TO_BIN: lambda x: bin(x)[2:],
        ConversionType.BIN_TO_DEC: lambda x: int(x, 2),
        ConversionType.DEC_TO_OCT: lambda x: oct(x)[2:],
        ConversionType.OCT_TO_DEC: lambda x: int(x, 8),
        ConversionType.DEC_TO_HEX: lambda x: hex(x)[2:],
        ConversionType.HEX_TO_DEC: lambda x: int(x, 16)
    }

    @staticmethod
    def convert(num, conversion_type):
        if conversion_type not in NumberConverter._CONVERSION_MAP:
            raise ValueError("Invalid conversion type")
        return NumberConverter._CONVERSION_MAP[conversion_type](num)

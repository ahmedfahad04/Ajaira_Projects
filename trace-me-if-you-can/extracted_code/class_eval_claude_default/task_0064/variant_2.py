from functools import partial

class NumberConverter:
    @staticmethod
    def _convert_from_decimal(decimal_num, converter_func):
        return converter_func(decimal_num)[2:]
    
    @staticmethod
    def _convert_to_decimal(num_str, base):
        return int(num_str, base)
    
    decimal_to_binary = partial(_convert_from_decimal.__func__, converter_func=bin)
    binary_to_decimal = partial(_convert_to_decimal.__func__, base=2)
    
    decimal_to_octal = partial(_convert_from_decimal.__func__, converter_func=oct)
    octal_to_decimal = partial(_convert_to_decimal.__func__, base=8)
    
    decimal_to_hex = partial(_convert_from_decimal.__func__, converter_func=hex)
    hex_to_decimal = partial(_convert_to_decimal.__func__, base=16)

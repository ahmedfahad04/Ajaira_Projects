import operator
from typing import Callable, Tuple

class NumberConverter:
    _CONVERTERS = {
        2: (bin, int),
        8: (oct, int),
        16: (hex, int)
    }
    
    @classmethod
    def _get_converter_pair(cls, base: int) -> Tuple[Callable, Callable]:
        return cls._CONVERTERS[base]
    
    @classmethod
    def _decimal_to_base(cls, decimal_num: int, base: int) -> str:
        to_base_func, _ = cls._get_converter_pair(base)
        return to_base_func(decimal_num)[2:]
    
    @classmethod
    def _base_to_decimal(cls, num_str: str, base: int) -> int:
        _, from_base_func = cls._get_converter_pair(base)
        return from_base_func(num_str, base)
    
    decimal_to_binary = staticmethod(lambda x: NumberConverter._decimal_to_base(x, 2))
    binary_to_decimal = staticmethod(lambda x: NumberConverter._base_to_decimal(x, 2))
    decimal_to_octal = staticmethod(lambda x: NumberConverter._decimal_to_base(x, 8))
    octal_to_decimal = staticmethod(lambda x: NumberConverter._base_to_decimal(x, 8))
    decimal_to_hex = staticmethod(lambda x: NumberConverter._decimal_to_base(x, 16))
    hex_to_decimal = staticmethod(lambda x: NumberConverter._base_to_decimal(x, 16))

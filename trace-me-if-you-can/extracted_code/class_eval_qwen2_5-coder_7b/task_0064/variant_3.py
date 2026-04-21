class NumberConverter:
    DEC_TO_BIN = lambda x: bin(x)[2:]
    BIN_TO_DEC = lambda x: int(x, 2)
    DEC_TO_OCT = lambda x: oct(x)[2:]
    OCT_TO_DEC = lambda x: int(x, 8)
    DEC_TO_HEX = lambda x: hex(x)[2:]
    HEX_TO_DEC = lambda x: int(x, 16)

    @staticmethod
    def convert(num, from_base, to_base):
        conversion_func = getattr(NumberConverter, f'{from_base.upper()}_{to_base.upper()}')
        return conversion_func(num)

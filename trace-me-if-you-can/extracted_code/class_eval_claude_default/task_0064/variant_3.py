class NumberConverter:
    def __init__(self):
        self.conversion_map = {
            ('decimal', 'binary'): lambda x: bin(x)[2:],
            ('binary', 'decimal'): lambda x: int(x, 2),
            ('decimal', 'octal'): lambda x: oct(x)[2:],
            ('octal', 'decimal'): lambda x: int(x, 8),
            ('decimal', 'hex'): lambda x: hex(x)[2:],
            ('hex', 'decimal'): lambda x: int(x, 16)
        }
    
    def convert(self, value, from_base, to_base):
        return self.conversion_map[(from_base, to_base)](value)
    
    @staticmethod
    def decimal_to_binary(decimal_num):
        converter = NumberConverter()
        return converter.convert(decimal_num, 'decimal', 'binary')
    
    @staticmethod
    def binary_to_decimal(binary_num):
        converter = NumberConverter()
        return converter.convert(binary_num, 'binary', 'decimal')
    
    @staticmethod
    def decimal_to_octal(decimal_num):
        converter = NumberConverter()
        return converter.convert(decimal_num, 'decimal', 'octal')
    
    @staticmethod
    def octal_to_decimal(octal_num):
        converter = NumberConverter()
        return converter.convert(octal_num, 'octal', 'decimal')
    
    @staticmethod
    def decimal_to_hex(decimal_num):
        converter = NumberConverter()
        return converter.convert(decimal_num, 'decimal', 'hex')
    
    @staticmethod
    def hex_to_decimal(hex_num):
        converter = NumberConverter()
        return converter.convert(hex_num, 'hex', 'decimal')

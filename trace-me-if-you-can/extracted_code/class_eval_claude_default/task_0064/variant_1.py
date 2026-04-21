class NumberConverter:
    BASE_FUNCTIONS = {
        'binary': (bin, 2),
        'octal': (oct, 8),
        'hex': (hex, 16)
    }
    
    @classmethod
    def convert_from_decimal(cls, decimal_num, base_name):
        func, _ = cls.BASE_FUNCTIONS[base_name]
        return func(decimal_num)[2:]
    
    @classmethod
    def convert_to_decimal(cls, num_str, base_name):
        _, base = cls.BASE_FUNCTIONS[base_name]
        return int(num_str, base)
    
    @staticmethod
    def decimal_to_binary(decimal_num):
        return NumberConverter.convert_from_decimal(decimal_num, 'binary')
    
    @staticmethod
    def binary_to_decimal(binary_num):
        return NumberConverter.convert_to_decimal(binary_num, 'binary')
    
    @staticmethod
    def decimal_to_octal(decimal_num):
        return NumberConverter.convert_from_decimal(decimal_num, 'octal')
    
    @staticmethod
    def octal_to_decimal(octal_num):
        return NumberConverter.convert_to_decimal(octal_num, 'octal')
    
    @staticmethod
    def decimal_to_hex(decimal_num):
        return NumberConverter.convert_from_decimal(decimal_num, 'hex')
    
    @staticmethod
    def hex_to_decimal(hex_num):
        return NumberConverter.convert_to_decimal(hex_num, 'hex')

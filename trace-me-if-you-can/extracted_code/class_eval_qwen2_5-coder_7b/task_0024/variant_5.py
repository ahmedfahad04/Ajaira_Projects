class ComplexOperations:
    @staticmethod
    def add_complex_numbers(c1, c2):
        return complex(c1.real + c2.real, c1.imag + c2.imag)
    
    @staticmethod
    def subtract_complex_numbers(c1, c2):
        return complex(c1.real - c2.real, c1.imag - c2.imag)
    
    @staticmethod
    def multiply_complex_numbers(c1, c2):
        real = c1.real * c2.real - c1.imag * c2.imag
        imaginary = c1.real * c2.imag + c1.imag * c2.real
        return complex(real, imaginary)
    
    @staticmethod
    def divide_complex_numbers(c1, c2):
        denominator = c2.real**2 + c2.imag**2
        real = (c1.real * c2.real + c1.imag * c2.imag) / denominator
        imaginary = (c1.imag * c2.real - c1.real * c2.imag) / denominator
        return complex(real, imaginary)

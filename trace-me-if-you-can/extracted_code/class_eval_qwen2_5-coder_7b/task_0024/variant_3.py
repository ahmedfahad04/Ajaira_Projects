class MathOperations:
    @staticmethod
    def sum_complex(c1, c2):
        return complex(c1.real + c2.real, c1.imag + c2.imag)
    
    @staticmethod
    def diff_complex(c1, c2):
        return complex(c1.real - c2.real, c1.imag - c2.imag)
    
    @staticmethod
    def prod_complex(c1, c2):
        real = c1.real * c2.real - c1.imag * c2.imag
        imaginary = c1.real * c2.imag + c1.imag * c2.real
        return complex(real, imaginary)
    
    @staticmethod
    def quot_complex(c1, c2):
        denominator = c2.real**2 + c2.imag**2
        real = (c1.real * c2.real + c1.imag * c2.imag) / denominator
        imaginary = (c1.imag * c2.real - c1.real * c2.imag) / denominator
        return complex(real, imaginary)

from functools import reduce
from operator import add, sub, mul, truediv

class ComplexCalculator:
    def __init__(self):
        pass

    @staticmethod
    def add(c1, c2):
        components = [(c1.real, c2.real), (c1.imag, c2.imag)]
        real, imaginary = map(lambda pair: reduce(add, pair), components)
        return complex(real, imaginary)
    
    @staticmethod
    def subtract(c1, c2):
        components = [(c1.real, c2.real), (c1.imag, c2.imag)]
        real, imaginary = map(lambda pair: reduce(sub, pair), components)
        return complex(real, imaginary)
    
    @staticmethod
    def multiply(c1, c2):
        real = reduce(sub, [c1.real * c2.real, c1.imag * c2.imag])
        imaginary = reduce(add, [c1.real * c2.imag, c1.imag * c2.real])
        return complex(real, imaginary)
    
    @staticmethod
    def divide(c1, c2):
        denominator = reduce(add, [c2.real**2, c2.imag**2])
        real = reduce(truediv, [reduce(add, [c1.real * c2.real, c1.imag * c2.imag]), denominator])
        imaginary = reduce(truediv, [reduce(sub, [c1.imag * c2.real, c1.real * c2.imag]), denominator])
        return complex(real, imaginary)

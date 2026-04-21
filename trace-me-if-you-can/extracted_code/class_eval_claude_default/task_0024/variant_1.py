class ComplexCalculator:
    def __init__(self):
        self.operations = {
            'add': lambda c1, c2: complex(c1.real + c2.real, c1.imag + c2.imag),
            'subtract': lambda c1, c2: complex(c1.real - c2.real, c1.imag - c2.imag),
            'multiply': lambda c1, c2: complex(c1.real * c2.real - c1.imag * c2.imag, 
                                              c1.real * c2.imag + c1.imag * c2.real),
            'divide': lambda c1, c2: complex((c1.real * c2.real + c1.imag * c2.imag) / (c2.real**2 + c2.imag**2),
                                            (c1.imag * c2.real - c1.real * c2.imag) / (c2.real**2 + c2.imag**2))
        }

    def add(self, c1, c2):
        return self.operations['add'](c1, c2)
    
    def subtract(self, c1, c2):
        return self.operations['subtract'](c1, c2)
    
    def multiply(self, c1, c2):
        return self.operations['multiply'](c1, c2)
    
    def divide(self, c1, c2):
        return self.operations['divide'](c1, c2)

class ComplexCalculator:
    def __init__(self):
        pass

    @staticmethod
    def _compute_binary_operation(c1, c2, operation):
        if operation == 'add':
            return complex(c1.real + c2.real, c1.imag + c2.imag)
        elif operation == 'subtract':
            return complex(c1.real - c2.real, c1.imag - c2.imag)
        elif operation == 'multiply':
            real_part = c1.real * c2.real - c1.imag * c2.imag
            imag_part = c1.real * c2.imag + c1.imag * c2.real
            return complex(real_part, imag_part)
        elif operation == 'divide':
            denominator = c2.real**2 + c2.imag**2
            real_part = (c1.real * c2.real + c1.imag * c2.imag) / denominator
            imag_part = (c1.imag * c2.real - c1.real * c2.imag) / denominator
            return complex(real_part, imag_part)

    @staticmethod
    def add(c1, c2):
        return ComplexCalculator._compute_binary_operation(c1, c2, 'add')
    
    @staticmethod
    def subtract(c1, c2):
        return ComplexCalculator._compute_binary_operation(c1, c2, 'subtract')
    
    @staticmethod
    def multiply(c1, c2):
        return ComplexCalculator._compute_binary_operation(c1, c2, 'multiply')
    
    @staticmethod
    def divide(c1, c2):
        return ComplexCalculator._compute_binary_operation(c1, c2, 'divide')

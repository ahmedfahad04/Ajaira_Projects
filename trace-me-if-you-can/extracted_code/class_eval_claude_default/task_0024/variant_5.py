class ComplexCalculator:
    def __init__(self):
        pass

    @staticmethod
    def add(c1, c2):
        result_components = [getattr(c1, attr) + getattr(c2, attr) for attr in ['real', 'imag']]
        return complex(*result_components)
    
    @staticmethod
    def subtract(c1, c2):
        result_components = [getattr(c1, attr) - getattr(c2, attr) for attr in ['real', 'imag']]
        return complex(*result_components)
    
    @staticmethod
    def multiply(c1, c2):
        a, b, c, d = c1.real, c1.imag, c2.real, c2.imag
        multiplication_matrix = [[a * c - b * d], [a * d + b * c]]
        return complex(*[row[0] for row in multiplication_matrix])
    
    @staticmethod
    def divide(c1, c2):
        a, b, c, d = c1.real, c1.imag, c2.real, c2.imag
        denominator = c * c + d * d
        division_numerators = [a * c + b * d, b * c - a * d]
        return complex(*[num / denominator for num in division_numerators])

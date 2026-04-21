from math import pi, fabs

class TriCalculator:

    def __init__(self):
        pass

    def _deg_to_rad(self, degrees):
        return degrees * pi / 180

    def _compute_factorial_iterative(self, n):
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def cos(self, x):
        x_rad = self._deg_to_rad(x)
        approximation = 0
        sign = 1
        for term_index in range(50):
            power = 2 * term_index
            coefficient = sign / self._compute_factorial_iterative(power)
            approximation += coefficient * (x_rad ** power)
            sign *= -1
        return round(approximation, 10)

    def sin(self, x):
        x_rad = self._deg_to_rad(x)
        current_term = x_rad
        total = 0
        iteration = 1
        
        while fabs(current_term) >= 1e-15:
            total += current_term
            iteration += 1
            current_term = -current_term * x_rad * x_rad / ((2 * iteration - 1) * (2 * iteration - 2))
        
        return round(total, 10)

    def tan(self, x):
        cosine_value = self.cos(x)
        sine_value = self.sin(x)
        return round(sine_value / cosine_value, 10) if cosine_value != 0 else False

from math import pi, fabs

class TriCalculator:

    PRECISION_THRESHOLD = 1e-15
    ROUNDING_PRECISION = 10
    MAX_TAYLOR_TERMS = 50

    def __init__(self):
        self.precomputed_factorials = self._precompute_factorials(100)

    def _precompute_factorials(self, max_n):
        factorials = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            factorials[i] = factorials[i-1] * i
        return factorials

    def _to_radians(self, degrees):
        return degrees * pi / 180

    def cos(self, x):
        x_rad = self._to_radians(x)
        result = self._cosine_taylor_series(x_rad)
        return round(result, self.ROUNDING_PRECISION)

    def _cosine_taylor_series(self, x_rad):
        total = 0
        for k in range(self.MAX_TAYLOR_TERMS):
            power = 2 * k
            sign = (-1) ** k
            term = sign * (x_rad ** power) / self.precomputed_factorials[power]
            total += term
        return total

    def sin(self, x):
        x_rad = self._to_radians(x)
        result = self._sine_convergent_series(x_rad)
        return round(result, self.ROUNDING_PRECISION)

    def _sine_convergent_series(self, x_rad):
        accumulator = 0
        current_term = x_rad
        term_number = 1

        while fabs(current_term) >= self.PRECISION_THRESHOLD:
            accumulator += current_term
            term_number += 1
            current_term *= -x_rad * x_rad / ((2 * term_number - 1) * (2 * term_number - 2))

        return accumulator

    def tan(self, x):
        cos_value = self.cos(x)
        sin_value = self.sin(x)
        
        if cos_value == 0:
            return False
        
        return round(sin_value / cos_value, self.ROUNDING_PRECISION)

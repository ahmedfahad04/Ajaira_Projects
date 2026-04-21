from math import pi, fabs

class TriCalculator:

    def __init__(self):
        self._factorial_cache = {0: 1, 1: 1}

    def _get_factorial(self, n):
        if n not in self._factorial_cache:
            self._factorial_cache[n] = n * self._get_factorial(n - 1)
        return self._factorial_cache[n]

    def cos(self, x):
        x_rad = x * pi / 180
        series_sum = 0
        for i in range(50):
            term = ((-1) ** i) * (x_rad ** (2 * i)) / self._get_factorial(2 * i)
            series_sum += term
        return round(series_sum, 10)

    def sin(self, x):
        x_rad = x * pi / 180
        series_sum = 0
        for i in range(50):
            term = ((-1) ** i) * (x_rad ** (2 * i + 1)) / self._get_factorial(2 * i + 1)
            series_sum += term
        return round(series_sum, 10)

    def tan(self, x):
        cos_result = self.cos(x)
        if cos_result == 0:
            return False
        return round(self.sin(x) / cos_result, 10)

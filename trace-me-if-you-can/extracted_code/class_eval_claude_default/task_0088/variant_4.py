from math import pi, fabs
from functools import lru_cache

class TriCalculator:

    def __init__(self):
        pass

    @lru_cache(maxsize=None)
    def factorial(self, n):
        return 1 if n <= 1 else n * self.factorial(n - 1)

    def cos(self, x):
        x_radians = x / 180 * pi
        taylor_series = lambda x, terms: sum(
            ((-1) ** k) * (x ** (2 * k)) / self.factorial(2 * k) 
            for k in range(terms)
        )
        return round(taylor_series(x_radians, 50), 10)

    def sin(self, x):
        x_radians = x / 180 * pi
        
        def sin_series_generator(x_rad):
            term = x_rad
            n = 1
            while fabs(term) >= 1e-15:
                yield term
                n += 1
                term = -term * x_rad * x_rad / (2 * n - 1) / (2 * n - 2)
        
        return round(sum(sin_series_generator(x_radians)), 10)

    def tan(self, x):
        cos_x, sin_x = self.cos(x), self.sin(x)
        return round(sin_x / cos_x, 10) if cos_x != 0 else False

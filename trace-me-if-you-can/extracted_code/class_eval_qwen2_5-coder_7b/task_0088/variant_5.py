import math


class TrigonometricOperations:
    def __init__(self):
        pass

    def cosine_value(self, angle):
        return round(self.taylor_series(angle, 50), 10)

    def compute_factorial_value(self, value):
        product = 1
        for i in range(1, value + 1):
            product *= i
        return product

    def taylor_series(self, value, terms):
        radians = value * (math.pi / 180)
        result = 1
        for i in range(1, terms):
            sign = -1 if i % 2 != 0 else 1
            result += sign * (radians ** (2 * i)) / self.compute_factorial_value(2 * i)
        return result

    def sine_value(self, angle):
        radians = angle * (math.pi / 180)
        approximation = 0
        term = radians
        iteration = 1

        while abs(term) >= 1e-15:
            approximation += term
            iteration += 1
            term = -term * radians * radians / (2 * iteration - 1) / (2 * iteration - 2)
        return round(approximation, 10)

    def tangent_value(self, angle):
        if self.cosine_value(angle) != 0:
            return round(self.sine_value(angle) / self.cosine_value(angle), 10)
        else:
            return False

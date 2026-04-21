import math


class TriangleMath:
    def __init__(self):
        pass

    def cosine(self, angle):
        return round(self.taylor_series(angle, 50), 10)

    def compute_factorial(self, value):
        product = 1
        for i in range(1, value + 1):
            product *= i
        return product

    def taylor_series(self, value, terms):
        radians = value * (math.pi / 180)
        sum_series = 1
        sign = 1
        for k in range(1, terms):
            sign *= -1
            sum_series += sign * (radians ** (2 * k)) / self.compute_factorial(2 * k)
        return sum_series

    def sine(self, angle):
        radians = angle * (math.pi / 180)
        approximation = 0
        term = radians
        iteration = 1

        while abs(term) >= 1e-15:
            approximation += term
            iteration += 1
            term = -term * radians * radians / (2 * iteration - 1) / (2 * iteration - 2)
        return round(approximation, 10)

    def tangent(self, angle):
        if self.cosine(angle) != 0:
            return round(self.sine(angle) / self.cosine(angle), 10)
        else:
            return False

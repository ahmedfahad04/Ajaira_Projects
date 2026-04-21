import math


class CalculateTrigonometry:
    def __init__(self):
        pass

    def calculate_cos(self, angle):
        return round(self.calculate_taylor(angle, 50), 10)

    def calculate_factorial(self, number):
        product = 1
        for i in range(1, number + 1):
            product *= i
        return product

    def calculate_taylor(self, value, terms):
        radians = value * (math.pi / 180)
        result = 1
        sign = 1
        for i in range(1, terms):
            sign *= -1
            result += sign * (radians ** (2 * i)) / self.calculate_factorial(2 * i)
        return result

    def calculate_sin(self, angle):
        radians = angle * (math.pi / 180)
        approx = 0
        term = radians
        iteration = 1

        while abs(term) >= 1e-15:
            approx += term
            iteration += 1
            term = -term * radians * radians / (2 * iteration - 1) / (2 * iteration - 2)
        return round(approx, 10)

    def calculate_tan(self, angle):
        if self.calculate_cos(angle) != 0:
            return round(self.calculate_sin(angle) / self.calculate_cos(angle), 10)
        else:
            return False

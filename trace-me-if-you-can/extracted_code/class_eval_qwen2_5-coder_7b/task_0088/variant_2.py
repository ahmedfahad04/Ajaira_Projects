from math import pi, fabs


class TrigOperations:
    def __init__(self):
        pass

    def calc_cos(self, angle):
        return round(self.calc_taylor_series(angle, 50), 10)

    def calc_factorial(self, num):
        result = 1
        for i in range(1, num + 1):
            result *= i
        return result

    def calc_taylor_series(self, value, terms):
        radians = value * (pi / 180)
        approximation = 1
        for i in range(1, terms):
            sign = -1 if i % 2 != 0 else 1
            approximation += sign * (radians ** (2 * i)) / self.calc_factorial(2 * i)
        return approximation

    def calc_sin(self, angle):
        radians = angle * (pi / 180)
        result = 0
        term = radians
        count = 1

        while fabs(term) >= 1e-15:
            result += term
            count += 1
            term = -term * radians * radians / (2 * count - 1) / (2 * count - 2)
        return round(result, 10)

    def calc_tan(self, angle):
        if self.calc_cos(angle) != 0:
            return round(self.calc_sin(angle) / self.calc_cos(angle), 10)
        else:
            return False

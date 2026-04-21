from math import pi, fabs


class TrigFunctions:
    def __init__(self):
        pass

    def cos_value(self, angle):
        return round(self.taylor_sum(angle, 50), 10)

    def compute_fact(self, value):
        product = 1
        for i in range(1, value + 1):
            product *= i
        return product

    def taylor_sum(self, value, terms):
        radians = value * (pi / 180)
        sum_series = 1
        for i in range(1, terms):
            if i % 2 != 0:
                sum_series -= (radians ** (2 * i)) / self.compute_fact(2 * i)
            else:
                sum_series += (radians ** (2 * i)) / self.compute_fact(2 * i)
        return sum_series

    def sin_value(self, angle):
        radians = angle * (pi / 180)
        result = 0
        term = radians
        count = 1

        while fabs(term) >= 1e-15:
            result += term
            count += 1
            term = -term * radians * radians / (2 * count - 1) / (2 * count - 2)
        return round(result, 10)

    def tan_value(self, angle):
        if self.cos_value(angle) != 0:
            return round(self.sin_value(angle) / self.cos_value(angle), 10)
        else:
            return False

import itertools
from functools import reduce
from operator import mul


class ArrangementCalculator:
    def __init__(self, datas):
        self.datas = datas

    @staticmethod
    def count(n, m=None):
        m = n if m is None else m
        return reduce(mul, range(n - m + 1, n + 1), 1)

    @staticmethod
    def count_all(n):
        return sum(ArrangementCalculator.count(n, i) for i in range(1, n + 1))

    def select(self, m=None):
        m = len(self.datas) if m is None else m
        return [list(perm) for perm in itertools.permutations(self.datas, m)]

    def select_all(self):
        return [list(perm) for i in range(1, len(self.datas) + 1) 
                for perm in itertools.permutations(self.datas, i)]

    @staticmethod
    def factorial(n):
        return reduce(mul, range(1, n + 1), 1)

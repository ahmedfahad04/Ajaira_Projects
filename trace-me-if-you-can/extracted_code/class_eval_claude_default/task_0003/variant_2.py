import itertools
import math


class ArrangementCalculator:
    def __init__(self, datas):
        self.datas = datas

    @staticmethod
    def count(n, m=None):
        if m is None or n == m:
            return math.factorial(n)
        return math.factorial(n) // math.factorial(n - m)

    @staticmethod
    def count_all(n):
        return math.factorial(n + 1) - 1

    def select(self, m=None):
        target_m = len(self.datas) if m is None else m
        return list(map(list, itertools.permutations(self.datas, target_m)))

    def select_all(self):
        arrangements = []
        data_length = len(self.datas)
        for length in range(1, data_length + 1):
            arrangements.extend(map(list, itertools.permutations(self.datas, length)))
        return arrangements

    @staticmethod
    def factorial(n):
        return math.factorial(n)

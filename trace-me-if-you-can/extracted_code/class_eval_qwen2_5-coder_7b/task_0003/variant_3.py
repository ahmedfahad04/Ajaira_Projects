import math
from itertools import permutations

class CombinatoricsTool:
    def __init__(self, collection):
        self.collection = collection

    @staticmethod
    def calculate_combinations(n, m=None):
        if m is None or n == m:
            return math.factorial(n)
        else:
            return math.factorial(n) // math.factorial(n - m)

    @staticmethod
    def calculate_total_combinations(n):
        total = 0
        for i in range(1, n + 1):
            total += CombinatoricsTool.calculate_combinations(n, i)
        return total

    def get_combinations(self, m=None):
        if m is None:
            m = len(self.collection)
        combinations = []
        for selection in permutations(self.collection, m):
            combinations.append(list(selection))
        return combinations

    def get_all_combinations(self):
        combinations = []
        for i in range(1, len(self.collection) + 1):
            combinations.extend(self.get_combinations(i))
        return combinations

    @staticmethod
    def calculate_factorial(n):
        return math.factorial(n)

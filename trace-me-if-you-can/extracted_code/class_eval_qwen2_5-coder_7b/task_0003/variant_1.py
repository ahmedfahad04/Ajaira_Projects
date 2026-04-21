import math
from itertools import permutations

class ArrangementSystem:
    def __init__(self, items):
        self.items = items

    @staticmethod
    def calculate_arrangements(n, m=None):
        if m is None or n == m:
            return math.factorial(n)
        else:
            return math.factorial(n) // math.factorial(n - m)

    @staticmethod
    def calculate_total_arrangements(n):
        total = 0
        for i in range(1, n + 1):
            total += ArrangementSystem.calculate_arrangements(n, i)
        return total

    def pick(self, m=None):
        if m is None:
            m = len(self.items)
        results = []
        for combo in permutations(self.items, m):
            results.append(list(combo))
        return results

    def pick_all(self):
        results = []
        for i in range(1, len(self.items) + 1):
            results.extend(self.pick(i))
        return results

    @staticmethod
    def calculate_factorial(n):
        return math.factorial(n)

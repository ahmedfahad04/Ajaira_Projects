import math
from itertools import permutations

class ArrangementEngine:
    def __init__(self, data_items):
        self.data_items = data_items

    @staticmethod
    def arrange(n, m=None):
        if m is None or n == m:
            return math.factorial(n)
        else:
            return math.factorial(n) // math.factorial(n - m)

    @staticmethod
    def arrange_all(n):
        total = 0
        for i in range(1, n + 1):
            total += ArrangementEngine.arrange(n, i)
        return total

    def choose(self, m=None):
        if m is None:
            m = len(self.data_items)
        choices = []
        for choice in permutations(self.data_items, m):
            choices.append(list(choice))
        return choices

    def choose_all(self):
        choices = []
        for i in range(1, len(self.data_items) + 1):
            choices.extend(self.choose(i))
        return choices

    @staticmethod
    def calculate_factorial(n):
        return math.factorial(n)

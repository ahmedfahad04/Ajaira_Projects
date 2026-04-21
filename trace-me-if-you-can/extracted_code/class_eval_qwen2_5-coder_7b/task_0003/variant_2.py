import math
from itertools import permutations

class PermutationCounter:
    def __init__(self, elements):
        self.elements = elements

    @staticmethod
    def compute_permutations(n, m=None):
        if m is None or n == m:
            return math.factorial(n)
        else:
            return math.factorial(n) // math.factorial(n - m)

    @staticmethod
    def compute_total_permutations(n):
        total = 0
        for i in range(1, n + 1):
            total += PermutationCounter.compute_permutations(n, i)
        return total

    def generate_permutations(self, m=None):
        if m is None:
            m = len(self.elements)
        outcomes = []
        for selection in permutations(self.elements, m):
            outcomes.append(list(selection))
        return outcomes

    def generate_all_permutations(self):
        outcomes = []
        for i in range(1, len(self.elements) + 1):
            outcomes.extend(self.generate_permutations(i))
        return outcomes

    @staticmethod
    def compute_factorial(n):
        return math.factorial(n)

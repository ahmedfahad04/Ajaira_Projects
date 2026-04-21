import math
from itertools import permutations

class CombinatorialSolver:
    def __init__(self, set_items):
        self.set_items = set_items

    @staticmethod
    def count_permutations(n, m=None):
        if m is None or n == m:
            return math.factorial(n)
        else:
            return math.factorial(n) // math.factorial(n - m)

    @staticmethod
    def count_total_permutations(n):
        total = 0
        for i in range(1, n + 1):
            total += CombinatorialSolver.count_permutations(n, i)
        return total

    def find_permutations(self, m=None):
        if m is None:
            m = len(self.set_items)
        permutation_list = []
        for permutation in permutations(self.set_items, m):
            permutation_list.append(list(permutation))
        return permutation_list

    def find_all_permutations(self):
        permutation_list = []
        for i in range(1, len(self.set_items) + 1):
            permutation_list.extend(self.find_permutations(i))
        return permutation_list

    @staticmethod
    def calculate_factorial(n):
        return math.factorial(n)

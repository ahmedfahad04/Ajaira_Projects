import itertools


class ArrangementCalculator:
    def __init__(self, datas):
        self.datas = tuple(datas)  # Make immutable

    @classmethod
    def _compute_permutation_count(cls, n, m):
        """Compute P(n,m) = n!/(n-m)! using iterative multiplication"""
        if m == 0:
            return 1
        product = 1
        for i in range(n, n - m, -1):
            product *= i
        return product

    @staticmethod
    def count(n, m=None):
        effective_m = n if m is None else m
        return ArrangementCalculator._compute_permutation_count(n, effective_m)

    @staticmethod
    def count_all(n):
        accumulator = 0
        for m in range(1, n + 1):
            accumulator += ArrangementCalculator._compute_permutation_count(n, m)
        return accumulator

    def select(self, m=None):
        size = len(self.datas) if m is None else m
        result_list = []
        perm_iterator = itertools.permutations(self.datas, size)
        for permutation_tuple in perm_iterator:
            result_list.append(list(permutation_tuple))
        return result_list

    def select_all(self):
        combined_results = []
        data_count = len(self.datas)
        for arrangement_size in range(1, data_count + 1):
            combined_results.extend(self.select(arrangement_size))
        return combined_results

    @staticmethod
    def factorial(n):
        return ArrangementCalculator._compute_permutation_count(n, n) if n > 0 else 1

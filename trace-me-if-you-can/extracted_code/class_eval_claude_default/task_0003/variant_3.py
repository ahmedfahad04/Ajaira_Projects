import itertools


class ArrangementCalculator:
    def __init__(self, datas):
        self.datas = datas
        self._memo = {}

    def count(self, n, m=None):
        key = (n, m)
        if key in self._memo:
            return self._memo[key]
        
        if m is None or n == m:
            result = self.factorial(n)
        else:
            result = self.factorial(n) // self.factorial(n - m)
        
        self._memo[key] = result
        return result

    def count_all(self, n):
        total = 0
        i = 1
        while i <= n:
            total += self.count(n, i)
            i += 1
        return total

    def select(self, m=None):
        selection_size = m or len(self.datas)
        permutation_generator = itertools.permutations(self.datas, selection_size)
        return [list(perm) for perm in permutation_generator]

    def select_all(self):
        all_arrangements = []
        for size in range(1, len(self.datas) + 1):
            all_arrangements += self.select(size)
        return all_arrangements

    def factorial(self, n):
        if n in self._memo:
            return self._memo[n]
        
        if n <= 1:
            result = 1
        else:
            result = n * self.factorial(n - 1)
        
        self._memo[n] = result
        return result

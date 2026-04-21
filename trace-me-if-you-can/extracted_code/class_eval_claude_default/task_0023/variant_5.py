import math
from typing import List

class CombinationCalculator:
    def __init__(self, datas: List[str]):
        self.datas = datas
        self._memo = {}

    @staticmethod
    def count(n: int, m: int) -> int:
        if m == 0 or n == m:
            return 1
        return math.factorial(n) // (math.factorial(n - m) * math.factorial(m))

    @staticmethod
    def count_all(n: int) -> int:
        if n < 0 or n > 63:
            return False
        return (1 << n) - 1 if n != 63 else float("inf")

    def select(self, m: int) -> List[List[str]]:
        key = (tuple(self.datas), m)
        if key in self._memo:
            return self._memo[key]
        
        if m == 0:
            self._memo[key] = [[]]
            return [[]]
        
        if m > len(self.datas):
            self._memo[key] = []
            return []
        
        if m == len(self.datas):
            self._memo[key] = [self.datas[:]]
            return [self.datas[:]]
        
        # Take combinations that include first element
        with_first = []
        if len(self.datas) > 1:
            sub_calc = CombinationCalculator(self.datas[1:])
            sub_combos = sub_calc.select(m - 1)
            with_first = [[self.datas[0]] + combo for combo in sub_combos]
        
        # Take combinations that exclude first element
        without_first = []
        if len(self.datas) > 1:
            sub_calc = CombinationCalculator(self.datas[1:])
            without_first = sub_calc.select(m)
        
        result = with_first + without_first
        self._memo[key] = result
        return result

    def select_all(self) -> List[List[str]]:
        result = []
        for i in range(1, len(self.datas) + 1):
            result.extend(self.select(i))
        return result

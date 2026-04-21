import math
from typing import List

class CombinationCalculator:
    def __init__(self, datas: List[str]):
        self.datas = datas

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
        def generate_combinations(items, k):
            if k == 0:
                yield []
                return
            if not items:
                return
            
            first = items[0]
            rest = items[1:]
            
            # Include first element
            for combo in generate_combinations(rest, k - 1):
                yield [first] + combo
            
            # Exclude first element
            yield from generate_combinations(rest, k)
        
        return list(generate_combinations(self.datas, m))

    def select_all(self) -> List[List[str]]:
        result = []
        for i in range(1, len(self.datas) + 1):
            result.extend(self.select(i))
        return result

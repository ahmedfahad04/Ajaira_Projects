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
        result = []
        n = len(self.datas)
        
        def backtrack(start_idx, current_combination):
            if len(current_combination) == m:
                result.append(current_combination[:])
                return
            
            remaining_needed = m - len(current_combination)
            remaining_available = n - start_idx
            
            if remaining_available < remaining_needed:
                return
            
            for i in range(start_idx, n):
                current_combination.append(self.datas[i])
                backtrack(i + 1, current_combination)
                current_combination.pop()
        
        backtrack(0, [])
        return result

    def select_all(self) -> List[List[str]]:
        result = []
        for i in range(1, len(self.datas) + 1):
            result.extend(self.select(i))
        return result

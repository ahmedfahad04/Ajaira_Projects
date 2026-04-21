import math
    from typing import List

    class CombinationFinder:
        def __init__(self, collection: List[str]):
            self.collection = collection

        @staticmethod
        def calculate_combinations(n: int, k: int) -> int:
            if k == 0 or n == k:
                return 1
            return math.factorial(n) // (math.factorial(n - k) * math.factorial(k))

        @staticmethod
        def calculate_total_combinations(n: int) -> int:
            if n < 0 or n > 63:
                return False
            return (1 << n) - 1 if n != 63 else float("inf")

        def find_combinations(self, k: int) -> List[List[str]]:
            combinations = []
            self._find_combinations(0, [None] * k, 0, combinations)
            return combinations

        def find_all_combinations(self) -> List[List[str]]:
            combinations = []
            for i in range(1, len(self.collection) + 1):
                combinations.extend(self.find_combinations(i))
            return combinations

        def _find_combinations(self, start: int, combo: List[str], index: int, combinations: List[List[str]]):
            combo_len = len(combo)
            combo_index_count = index + 1
            if combo_index_count > combo_len:
                combinations.append(combo.copy())
                return

            for i in range(start, len(self.collection) + combo_index_count - combo_len):
                combo[index] = self.collection[i]
                self._find_combinations(i + 1, combo, index + 1, combinations)

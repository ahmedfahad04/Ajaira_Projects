import math
    from typing import List

    class Combinatorics:
        def __init__(self, elements: List[str]):
            self.elements = elements

        @staticmethod
        def comb(n: int, k: int) -> int:
            if k == 0 or n == k:
                return 1
            return math.factorial(n) // (math.factorial(n - k) * math.factorial(k))

        @staticmethod
        def comb_total(n: int) -> int:
            if n < 0 or n > 63:
                return False
            return (1 << n) - 1 if n != 63 else float("inf")

        def get_combinations(self, k: int) -> List[List[str]]:
            outcomes = []
            self._get_combinations(0, [None] * k, 0, outcomes)
            return outcomes

        def get_all_combinations(self) -> List[List[str]]:
            outcomes = []
            for i in range(1, len(self.elements) + 1):
                outcomes.extend(self.get_combinations(i))
            return outcomes

        def _get_combinations(self, start_index: int, current_combo: List[str], combo_index: int, outcomes: List[List[str]]):
            combo_length = len(current_combo)
            combo_count = combo_index + 1
            if combo_count > combo_length:
                outcomes.append(current_combo.copy())
                return

            for i in range(start_index, len(self.elements) + combo_count - combo_length):
                current_combo[combo_index] = self.elements[i]
                self._get_combinations(i + 1, current_combo, combo_index + 1, outcomes)

import math
    from typing import List

    class CombinationUtil:
        def __init__(self, elements: List[str]):
            self.elements = elements

        @staticmethod
        def combinatorial(n: int, k: int) -> int:
            if k == 0 or n == k:
                return 1
            return math.factorial(n) // (math.factorial(n - k) * math.factorial(k))

        @staticmethod
        def combinatorial_total(n: int) -> int:
            if n < 0 or n > 63:
                return False
            return (1 << n) - 1 if n != 63 else float("inf")

        def select_combinations(self, k: int) -> List[List[str]]:
            combinations = []
            self._select_combinations(0, [None] * k, 0, combinations)
            return combinations

        def select_all_combinations(self) -> List[List[str]]:
            combinations = []
            for i in range(1, len(self.elements) + 1):
                combinations.extend(self.select_combinations(i))
            return combinations

        def _select_combinations(self, current_index: int, combo_list: List[str], combo_pos: int, combinations: List[List[str]]):
            combo_length = len(combo_list)
            combo_pos_count = combo_pos + 1
            if combo_pos_count > combo_length:
                combinations.append(combo_list.copy())
                return

            for i in range(current_index, len(self.elements) + combo_pos_count - combo_length):
                combo_list[combo_pos] = self.elements[i]
                self._select_combinations(i + 1, combo_list, combo_pos + 1, combinations)

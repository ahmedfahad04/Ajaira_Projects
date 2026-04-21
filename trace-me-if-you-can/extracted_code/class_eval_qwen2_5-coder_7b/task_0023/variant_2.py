import math
    from typing import List

    class Combinator:
        def __init__(self, items: List[str]):
            self.items = items

        @staticmethod
        def compute_comb(n: int, k: int) -> int:
            if k == 0 or n == k:
                return 1
            return math.factorial(n) // (math.factorial(n - k) * math.factorial(k))

        @staticmethod
        def compute_total_comb(n: int) -> int:
            if n < 0 or n > 63:
                return False
            return (1 << n) - 1 if n != 63 else float("inf")

        def fetch_combinations(self, k: int) -> List[List[str]]:
            combinations = []
            self._fetch_combinations(0, [None] * k, 0, combinations)
            return combinations

        def fetch_all_combinations(self) -> List[List[str]]:
            combinations = []
            for i in range(1, len(self.items) + 1):
                combinations.extend(self.fetch_combinations(i))
            return combinations

        def _fetch_combinations(self, index: int, combo_list: List[str], combo_pos: int, combinations: List[List[str]]):
            combo_len = len(combo_list)
            combo_pos_count = combo_pos + 1
            if combo_pos_count > combo_len:
                combinations.append(combo_list.copy())
                return

            for i in range(index, len(self.items) + combo_pos_count - combo_len):
                combo_list[combo_pos] = self.items[i]
                self._fetch_combinations(i + 1, combo_list, combo_pos + 1, combinations)

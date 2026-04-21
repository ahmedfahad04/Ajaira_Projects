import math
    from typing import List

    class CombinationSelector:
        def __init__(self, data: List[str]):
            self.data = data

        @staticmethod
        def calculate_comb(n: int, k: int) -> int:
            if k == 0 or n == k:
                return 1
            return math.factorial(n) // (math.factorial(n - k) * math.factorial(k))

        @staticmethod
        def calculate_total_comb(n: int) -> int:
            if n < 0 or n > 63:
                return False
            return (1 << n) - 1 if n != 63 else float("inf")

        def compute_selections(self, k: int) -> List[List[str]]:
            selections = []
            self._compute_selections(0, [None] * k, 0, selections)
            return selections

        def compute_all_selections(self) -> List[List[str]]:
            selections = []
            for i in range(1, len(self.data) + 1):
                selections.extend(self.compute_selections(i))
            return selections

        def _compute_selections(self, current_index: int, selection_list: List[str], selection_index: int, selections: List[List[str]]):
            selection_length = len(selection_list)
            selection_count = selection_index + 1
            if selection_count > selection_length:
                selections.append(selection_list.copy())
                return

            for i in range(current_index, len(self.data) + selection_count - selection_length):
                selection_list[selection_index] = self.data[i]
                self._compute_selections(i + 1, selection_list, selection_index + 1, selections)

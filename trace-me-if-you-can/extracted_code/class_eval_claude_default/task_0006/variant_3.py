from functools import lru_cache

class AvgPartition:
    def __init__(self, lst, limit):
        self.lst = lst
        self.limit = limit

    @lru_cache(maxsize=None)
    def _calculate_bounds(self, index):
        base_size = len(self.lst) // self.limit
        extra_items = len(self.lst) % self.limit
        
        items_before = index * base_size + min(index, extra_items)
        current_size = base_size + (1 if index < extra_items else 0)
        
        return items_before, items_before + current_size
        
    def get(self, index):
        start, end = self._calculate_bounds(index)
        return self.lst[start:end]

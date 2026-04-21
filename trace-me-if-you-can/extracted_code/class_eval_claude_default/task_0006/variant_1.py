class AvgPartition:
    def __init__(self, lst, limit):
        self.lst = lst
        self.limit = limit
        self._size, self._remainder = divmod(len(lst), limit)

    def get(self, index):
        partition_size = self._size + (1 if index < self._remainder else 0)
        start = index * self._size + min(index, self._remainder)
        return self.lst[start:start + partition_size]

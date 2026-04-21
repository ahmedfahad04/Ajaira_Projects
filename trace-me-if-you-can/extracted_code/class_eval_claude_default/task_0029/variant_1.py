from collections import Counter
from functools import reduce
from operator import add


class DataStatistics:
    def mean(self, data):
        total = reduce(add, data, 0)
        return round(total / len(data), 2)

    def median(self, data):
        sorted_data = sorted(data)
        n = len(sorted_data)
        mid_index = n // 2
        
        if n % 2 == 0:
            return round((sorted_data[mid_index - 1] + sorted_data[mid_index]) / 2, 2)
        return sorted_data[mid_index]

    def mode(self, data):
        counter = Counter(data)
        max_count = max(counter.values())
        return list(filter(lambda x: counter[x] == max_count, counter.keys()))

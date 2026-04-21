from statistics import median
from collections import Counter


class DataInfo:
    def calculate_mean(self, data):
        return round(sum(data) / len(data), 2)

    def find_median(self, data):
        sorted_data = sorted(data)
        n = len(sorted_data)
        mid = n // 2
        if n % 2 == 0:
            return round((sorted_data[mid - 1] + sorted_data[mid]) / 2, 2)
        else:
            return sorted_data[mid]

    def determine_mode(self, data):
        counter = Counter(data)
        max_freq = max(counter.values())
        return [x for x, freq in counter.items() if freq == max_freq]

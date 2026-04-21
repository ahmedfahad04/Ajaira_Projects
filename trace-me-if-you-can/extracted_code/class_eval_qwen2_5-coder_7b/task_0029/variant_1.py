from statistics import mean
from collections import Counter


class DataAnalysis:
    def average(self, values):
        return round(mean(values), 2)

    def mid_value(self, values):
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            return round((sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2, 2)
        else:
            return sorted_values[n // 2]

    def most_common(self, values):
        counter = Counter(values)
        max_count = max(counter.values())
        return [x for x, count in counter.items() if count == max_count]

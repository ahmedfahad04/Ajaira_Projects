from statistics import mean
from collections import Counter


class StatisticCalculator:
    def avg(self, data):
        return round(mean(data), 2)

    def calc_median(self, data):
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 0:
            return round((sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2, 2)
        else:
            return sorted_data[n // 2]

    def get_mode(self, data):
        counter = Counter(data)
        mode_count = max(counter.values())
        return [x for x, count in counter.items() if count == mode_count]

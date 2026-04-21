from statistics import mean
from collections import Counter

class DataStats:
    def calculate_mean(self, values):
        return round(mean(values), 2)

    def find_median(self, numbers):
        sorted_nums = sorted(numbers)
        length = len(sorted_nums)
        if length % 2 == 0:
            mid = length // 2
            return round((sorted_nums[mid - 1] + sorted_nums[mid]) / 2, 2)
        else:
            mid = length // 2
            return sorted_nums[mid]

    def determine_mode(self, data):
        counts = Counter(data)
        max_count = max(counts.values())
        modes = [x for x, count in counts.items() if count == max_count]
        return modes

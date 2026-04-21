from collections import Counter


class DataStatistics:
    def mean(self, data):
        return round(sum(x for x in data) / len(data), 2)

    def median(self, data):
        def _get_median_value(sorted_list, size):
            half = size // 2
            if size % 2 == 0:
                yield round((sorted_list[half - 1] + sorted_list[half]) / 2, 2)
            else:
                yield sorted_list[half]
        
        sorted_data = sorted(data)
        return next(_get_median_value(sorted_data, len(sorted_data)))

    def mode(self, data):
        counts = Counter(data)
        peak_count = max(counts.values())
        return [val for val, count in counts.items() if count == peak_count]

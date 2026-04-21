from collections import Counter
import statistics


class DataStatistics:
    def mean(self, data):
        data_iter = iter(data)
        total = 0
        count = 0
        for value in data_iter:
            total += value
            count += 1
        return round(total / count, 2)

    def median(self, data):
        sorted_data = sorted(data)
        length = len(sorted_data)
        center = length // 2
        
        return (round((sorted_data[center - 1] + sorted_data[center]) / 2, 2) 
                if length % 2 == 0 
                else sorted_data[center])

    def mode(self, data):
        frequency_map = Counter(data)
        highest_frequency = max(frequency_map.values())
        modes = []
        for value, freq in frequency_map.items():
            if freq == highest_frequency:
                modes.append(value)
        return modes

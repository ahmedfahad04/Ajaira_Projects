from collections import Counter


class DataStatistics:
    def _calculate_sum(self, data):
        return sum(data)
    
    def _get_sorted_data(self, data):
        return sorted(data)
    
    def _find_middle_indices(self, length):
        return length // 2, length % 2 == 0

    def mean(self, data):
        total = self._calculate_sum(data)
        return round(total / len(data), 2)

    def median(self, data):
        sorted_data = self._get_sorted_data(data)
        middle_idx, is_even = self._find_middle_indices(len(sorted_data))
        
        if is_even:
            return round((sorted_data[middle_idx - 1] + sorted_data[middle_idx]) / 2, 2)
        else:
            return sorted_data[middle_idx]

    def mode(self, data):
        frequency_counter = Counter(data)
        maximum_frequency = max(frequency_counter.values())
        return [element for element, freq in frequency_counter.items() 
                if freq == maximum_frequency]

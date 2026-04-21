from collections import Counter


class DataStatistics:
    def mean(self, data):
        accumulator = 0
        for i in range(len(data)):
            accumulator += data[i]
        return round(accumulator / len(data), 2)

    def median(self, data):
        sorted_data = sorted(data)
        data_length = len(sorted_data)
        midpoint = data_length // 2
        
        if data_length & 1:  # odd length (bitwise check)
            return sorted_data[midpoint]
        else:  # even length
            left_middle = sorted_data[midpoint - 1]
            right_middle = sorted_data[midpoint]
            return round((left_middle + right_middle) / 2, 2)

    def mode(self, data):
        element_counts = Counter(data)
        max_occurrence = max(element_counts.values())
        modal_values = []
        
        for element in element_counts:
            if element_counts[element] == max_occurrence:
                modal_values.append(element)
        
        return modal_values

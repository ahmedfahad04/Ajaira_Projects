import math
from functools import reduce
from operator import add

class Statistics3:
    @staticmethod
    def median(data):
        sorted_list = sorted(data)
        length = len(sorted_list)
        middle = length // 2
        
        if length & 1:  # odd length
            return sorted_list[middle]
        return (sorted_list[middle - 1] + sorted_list[middle]) * 0.5

    @staticmethod
    def mode(data):
        def count_occurrences(acc, val):
            acc[val] = acc.get(val, 0) + 1
            return acc
        
        counts = reduce(count_occurrences, data, {})
        max_frequency = max(counts.values())
        return [k for k, v in counts.items() if v == max_frequency]

    @staticmethod
    def correlation(x, y):
        n = len(x)
        if n == 0:
            return None
            
        def compute_stats(values):
            total = reduce(add, values)
            mean = total / n
            sum_squares = reduce(add, (v * v for v in values))
            return total, mean, sum_squares
        
        sum_x, mean_x, sum_x_sq = compute_stats(x)
        sum_y, mean_y, sum_y_sq = compute_stats(y)
        sum_xy = reduce(add, (xi * yi for xi, yi in zip(x, y)))
        
        numerator = n * sum_xy - sum_x * sum_y
        variance_x = n * sum_x_sq - sum_x * sum_x
        variance_y = n * sum_y_sq - sum_y * sum_y
        denominator = math.sqrt(variance_x * variance_y)
        
        return numerator / denominator if denominator != 0 else None

    @staticmethod
    def mean(data):
        return reduce(add, data) / len(data) if data else None

    @staticmethod
    def correlation_matrix(data):
        def extract_column(col_index):
            return [row[col_index] for row in data]
        
        num_columns = len(data[0])
        matrix = []
        
        for i in range(num_columns):
            row = []
            col_i = extract_column(i)
            for j in range(num_columns):
                col_j = extract_column(j)
                row.append(Statistics3.correlation(col_i, col_j))
            matrix.append(row)
        
        return matrix

    @staticmethod
    def standard_deviation(data):
        n = len(data)
        if n < 2:
            return None
            
        mean_val = Statistics3.mean(data)
        squared_deviations = [(x - mean_val) ** 2 for x in data]
        variance = reduce(add, squared_deviations) / (n - 1)
        return math.sqrt(variance)

    @staticmethod
    def z_score(data):
        mean_val = Statistics3.mean(data)
        std_val = Statistics3.standard_deviation(data)
        
        if not std_val or std_val == 0:
            return None
            
        return list(map(lambda x: (x - mean_val) / std_val, data))

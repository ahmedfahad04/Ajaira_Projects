import math
from collections import Counter

class Statistics3:
    @staticmethod
    def median(data):
        sorted_data = sorted(data)
        n = len(sorted_data)
        mid = n // 2
        return sorted_data[mid] if n % 2 else (sorted_data[mid-1] + sorted_data[mid]) / 2

    @staticmethod
    def mode(data):
        counter = Counter(data)
        max_count = counter.most_common(1)[0][1]
        return [value for value, count in counter.items() if count == max_count]

    @staticmethod
    def correlation(x, y):
        n = len(x)
        if n == 0:
            return None
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xx = sum(xi * xi for xi in x)
        sum_yy = sum(yi * yi for yi in y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y))
        
        return numerator / denominator if denominator != 0 else None

    @staticmethod
    def mean(data):
        return sum(data) / len(data) if data else None

    @staticmethod
    def correlation_matrix(data):
        num_cols = len(data[0])
        columns = [[row[i] for row in data] for i in range(num_cols)]
        
        return [[Statistics3.correlation(columns[i], columns[j]) 
                for j in range(num_cols)] 
               for i in range(num_cols)]

    @staticmethod
    def standard_deviation(data):
        if len(data) < 2:
            return None
        mean_val = Statistics3.mean(data)
        return math.sqrt(sum((x - mean_val) ** 2 for x in data) / (len(data) - 1))

    @staticmethod
    def z_score(data):
        mean_val = Statistics3.mean(data)
        std_dev = Statistics3.standard_deviation(data)
        return [(x - mean_val) / std_dev for x in data] if std_dev and std_dev != 0 else None

import math

class Statistics3:
    @staticmethod
    def median(data):
        data_copy = data[:]
        data_copy.sort()
        n = len(data_copy)
        
        if n % 2:
            return data_copy[n >> 1]
        else:
            mid_idx = n >> 1
            return (data_copy[mid_idx - 1] + data_copy[mid_idx]) / 2.0

    @staticmethod
    def mode(data):
        value_counts = {}
        max_count = 0
        
        for val in data:
            count = value_counts.get(val, 0) + 1
            value_counts[val] = count
            if count > max_count:
                max_count = count
        
        return [val for val, count in value_counts.items() if count == max_count]

    @staticmethod
    def correlation(x, y):
        n = len(x)
        if n <= 1:
            return None
        
        # Calculate means
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # Calculate correlation components in single pass
        numerator = 0
        sum_sq_x = 0
        sum_sq_y = 0
        
        for i in range(n):
            diff_x = x[i] - mean_x
            diff_y = y[i] - mean_y
            numerator += diff_x * diff_y
            sum_sq_x += diff_x * diff_x
            sum_sq_y += diff_y * diff_y
        
        denominator = math.sqrt(sum_sq_x * sum_sq_y)
        return numerator / denominator if denominator > 0 else None

    @staticmethod
    def mean(data):
        n = len(data)
        return sum(data) / n if n > 0 else None

    @staticmethod
    def correlation_matrix(data):
        rows, cols = len(data), len(data[0])
        
        # Pre-extract all columns for efficiency
        column_data = []
        for col in range(cols):
            column_data.append([data[row][col] for row in range(rows)])
        
        # Build correlation matrix
        result = []
        for i in range(cols):
            row = []
            for j in range(cols):
                corr = Statistics3.correlation(column_data[i], column_data[j])
                row.append(corr)
            result.append(row)
        
        return result

    @staticmethod
    def standard_deviation(data):
        n = len(data)
        if n < 2:
            return None
        
        mean_value = Statistics3.mean(data)
        sum_squared_diff = 0
        
        for value in data:
            diff = value - mean_value
            sum_squared_diff += diff * diff
        
        return math.sqrt(sum_squared_diff / (n - 1))

    @staticmethod
    def z_score(data):
        mean_val = Statistics3.mean(data)
        if mean_val is None:
            return None
            
        std_dev = Statistics3.standard_deviation(data)
        if std_dev is None or std_dev == 0:
            return None
        
        z_scores = []
        for value in data:
            z_scores.append((value - mean_val) / std_dev)
        
        return z_scores

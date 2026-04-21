import math

class Statistics3:
    @staticmethod
    def median(data):
        n = len(data)
        sorted_data = sorted(data)
        return (sorted_data[(n-1)//2] + sorted_data[n//2]) / 2

    @staticmethod
    def mode(data):
        frequency_map = {}
        for item in data:
            frequency_map[item] = frequency_map.setdefault(item, 0) + 1
        
        highest_frequency = max(frequency_map.values())
        return [key for key, freq in frequency_map.items() if freq == highest_frequency]

    @staticmethod
    def correlation(x, y):
        n = len(x)
        if n == 0:
            return None
            
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        covariance = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        x_variance = sum((x[i] - x_mean) ** 2 for i in range(n))
        y_variance = sum((y[i] - y_mean) ** 2 for i in range(n))
        
        denominator = math.sqrt(x_variance * y_variance)
        return covariance / denominator if denominator > 0 else None

    @staticmethod
    def mean(data):
        return None if not data else sum(data) / len(data)

    @staticmethod
    def correlation_matrix(data):
        num_features = len(data[0])
        result = []
        
        for row_idx in range(num_features):
            correlation_row = []
            col1 = [data[i][row_idx] for i in range(len(data))]
            
            for col_idx in range(num_features):
                col2 = [data[i][col_idx] for i in range(len(data))]
                correlation_row.append(Statistics3.correlation(col1, col2))
            
            result.append(correlation_row)
        return result

    @staticmethod
    def standard_deviation(data):
        n = len(data)
        if n < 2:
            return None
        
        avg = Statistics3.mean(data)
        sum_squared_deviations = sum((value - avg) ** 2 for value in data)
        return math.sqrt(sum_squared_deviations / (n - 1))

    @staticmethod
    def z_score(data):
        avg = Statistics3.mean(data)
        std = Statistics3.standard_deviation(data)
        
        if std is None or std == 0:
            return None
        
        return [(value - avg) / std for value in data]

import math

class DataStatistics4:

    @staticmethod
    def correlation_coefficient(data1, data2):
        # Use generator expressions with zip for memory efficiency
        pairs = list(zip(data1, data2))
        n = len(pairs)
        
        mean1 = sum(x for x, _ in pairs) / n
        mean2 = sum(y for _, y in pairs) / n
        
        # Calculate all components using the paired data
        covariance = sum((x - mean1) * (y - mean2) for x, y in pairs)
        variance1 = sum((x - mean1) ** 2 for x, _ in pairs)
        variance2 = sum((y - mean2) ** 2 for _, y in pairs)
        
        denominator = math.sqrt(variance1 * variance2)
        return covariance / denominator if denominator != 0 else 0
    
    @staticmethod
    def skewness(data):
        n = len(data)
        
        # Create iterator for efficient processing
        data_iter = iter(data)
        mean = sum(data_iter) / n
        
        # Reset iterator and calculate variance
        standardized_values = [(x - mean) for x in data]
        variance = sum(val ** 2 for val in standardized_values) / n
        
        if variance == 0:
            return 0
            
        std_dev = math.sqrt(variance)
        third_moment_sum = sum(val ** 3 for val in standardized_values)
        
        return (third_moment_sum * n) / ((n - 1) * (n - 2) * std_dev ** 3)
    
    @staticmethod
    def kurtosis(data):
        n = len(data)
        
        # Transform data to centered values first
        mean = sum(data) / n
        centered = [x - mean for x in data]
        
        # Calculate moments from centered data
        second_moment = sum(x ** 2 for x in centered) / n
        
        if second_moment == 0:
            return math.nan
            
        fourth_moment = sum(x ** 4 for x in centered) / n
        excess_kurtosis = (fourth_moment / (second_moment ** 2)) - 3
        
        return excess_kurtosis
    
    @staticmethod
    def pdf(data, mu, sigma):
        # Use map for functional approach
        def normal_pdf_point(x):
            standardized = (x - mu) / sigma
            return math.exp(-0.5 * standardized ** 2) / (sigma * math.sqrt(2 * math.pi))
        
        return list(map(normal_pdf_point, data))

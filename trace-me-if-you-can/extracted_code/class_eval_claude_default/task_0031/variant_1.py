import math
from functools import reduce

class DataStatistics4:

    @staticmethod
    def correlation_coefficient(data1, data2):
        n = len(data1)
        sum1, sum2 = map(sum, [data1, data2])
        mean1, mean2 = sum1 / n, sum2 / n
        
        sum_xy = reduce(lambda acc, pair: acc + (pair[0] - mean1) * (pair[1] - mean2), zip(data1, data2), 0)
        sum_x2 = reduce(lambda acc, x: acc + (x - mean1) ** 2, data1, 0)
        sum_y2 = reduce(lambda acc, y: acc + (y - mean2) ** 2, data2, 0)
        
        denominator = math.sqrt(sum_x2 * sum_y2)
        return sum_xy / denominator if denominator != 0 else 0
    
    @staticmethod
    def skewness(data):
        n = len(data)
        if n < 3:
            return 0
            
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n
        
        if variance == 0:
            return 0
            
        std_dev = math.sqrt(variance)
        third_moment = sum((x - mean) ** 3 for x in data)
        
        return (third_moment * n) / ((n - 1) * (n - 2) * std_dev ** 3)
    
    @staticmethod
    def kurtosis(data):
        n = len(data)
        mean = sum(data) / n
        
        deviations = [x - mean for x in data]
        variance = sum(dev ** 2 for dev in deviations) / n
        
        if variance == 0:
            return math.nan
            
        fourth_moment = sum(dev ** 4 for dev in deviations) / n
        return (fourth_moment / (variance ** 2)) - 3
    
    @staticmethod
    def pdf(data, mu, sigma):
        coefficient = 1 / (sigma * math.sqrt(2 * math.pi))
        return [coefficient * math.exp(-0.5 * ((x - mu) / sigma) ** 2) for x in data]

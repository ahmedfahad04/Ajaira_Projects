import math

class DataStatistics4:

    @staticmethod
    def correlation_coefficient(data1, data2):
        n = len(data1)
        
        # Calculate sums in single pass
        sx = sy = sxx = syy = sxy = 0
        for x, y in zip(data1, data2):
            sx += x
            sy += y
            sxx += x * x
            syy += y * y
            sxy += x * y
        
        # Calculate correlation using computational formula
        numerator = n * sxy - sx * sy
        denominator_x = n * sxx - sx * sx
        denominator_y = n * syy - sy * sy
        
        denominator = math.sqrt(denominator_x * denominator_y)
        return numerator / denominator if denominator != 0 else 0
    
    @staticmethod
    def skewness(data):
        n = len(data)
        if n < 3:
            return 0
            
        # Single pass calculation
        s1 = s2 = s3 = 0
        for x in data:
            s1 += x
            s2 += x * x
            s3 += x * x * x
        
        mean = s1 / n
        variance = (s2 - n * mean * mean) / n
        
        if variance == 0:
            return 0
            
        std_dev = math.sqrt(variance)
        third_central = s3 - 3 * mean * s2 + 3 * mean * mean * s1 - n * mean ** 3
        
        return (third_central * n) / ((n - 1) * (n - 2) * std_dev ** 3)
    
    @staticmethod
    def kurtosis(data):
        n = len(data)
        
        # Accumulate moments in one pass
        moments = [0, 0, 0, 0, 0]  # s0, s1, s2, s3, s4
        for x in data:
            power = 1
            for i in range(5):
                moments[i] += power
                power *= x
        
        mean = moments[1] / n
        variance = (moments[2] - n * mean ** 2) / n
        
        if variance == 0:
            return math.nan
            
        fourth_central = (moments[4] - 4 * mean * moments[3] + 
                         6 * mean ** 2 * moments[2] - 
                         4 * mean ** 3 * moments[1] + 
                         n * mean ** 4)
        
        return (fourth_central / n) / (variance ** 2) - 3
    
    @staticmethod
    def pdf(data, mu, sigma):
        # Precompute constants
        sqrt_2pi = math.sqrt(2 * math.pi)
        coefficient = 1 / (sigma * sqrt_2pi)
        variance = sigma * sigma
        
        result = []
        for x in data:
            diff = x - mu
            exponent = -(diff * diff) / (2 * variance)
            result.append(coefficient * math.exp(exponent))
        
        return result

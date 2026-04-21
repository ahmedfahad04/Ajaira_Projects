import math

class DataStatistics4:

    @staticmethod
    def correlation_coefficient(data1, data2):
        def calculate_moments(arr1, arr2):
            n = len(arr1)
            m1, m2 = sum(arr1) / n, sum(arr2) / n
            
            covariance = sum((arr1[i] - m1) * (arr2[i] - m2) for i in range(n))
            var1 = sum((arr1[i] - m1) ** 2 for i in range(n))
            var2 = sum((arr2[i] - m2) ** 2 for i in range(n))
            
            return covariance, math.sqrt(var1 * var2)
        
        numerator, denominator = calculate_moments(data1, data2)
        return numerator / denominator if denominator != 0 else 0
    
    @staticmethod
    def skewness(data):
        def compute_central_moments(values):
            n = len(values)
            mean_val = sum(values) / n
            
            second_moment = sum((x - mean_val) ** 2 for x in values) / n
            third_moment = sum((x - mean_val) ** 3 for x in values)
            
            return mean_val, second_moment, third_moment
        
        n = len(data)
        mean_val, variance, third_raw = compute_central_moments(data)
        
        std_dev = math.sqrt(variance)
        if std_dev == 0:
            return 0
            
        return (third_raw * n) / ((n - 1) * (n - 2) * std_dev ** 3)
    
    @staticmethod
    def kurtosis(data):
        def moment_calculator(dataset, center, power):
            return sum((x - center) ** power for x in dataset) / len(dataset)
        
        n = len(data)
        mean_val = sum(data) / n
        
        second_moment = moment_calculator(data, mean_val, 2)
        fourth_moment = moment_calculator(data, mean_val, 4)
        
        if second_moment == 0:
            return math.nan
            
        return (fourth_moment / (second_moment ** 2)) - 3
    
    @staticmethod
    def pdf(data, mu, sigma):
        def gaussian_density(x, mean, std):
            exponent = -0.5 * ((x - mean) / std) ** 2
            return math.exp(exponent) / (std * math.sqrt(2 * math.pi))
        
        return [gaussian_density(point, mu, sigma) for point in data]

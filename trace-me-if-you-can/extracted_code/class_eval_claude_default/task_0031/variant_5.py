import math

class DataStatistics4:

    @staticmethod
    def correlation_coefficient(data1, data2):
        # Matrix-style computation approach
        n = len(data1)
        
        # Calculate means
        mean_vector = [sum(data1) / n, sum(data2) / n]
        
        # Build deviation matrices
        deviations = [[data1[i] - mean_vector[0], data2[i] - mean_vector[1]] for i in range(n)]
        
        # Compute covariance and variances using dot products
        covariance = sum(dev[0] * dev[1] for dev in deviations)
        var1 = sum(dev[0] ** 2 for dev in deviations)
        var2 = sum(dev[1] ** 2 for dev in deviations)
        
        correlation = covariance / math.sqrt(var1 * var2) if var1 * var2 > 0 else 0
        return correlation
    
    @staticmethod
    def skewness(data):
        # Statistical moments approach with error handling
        try:
            n = len(data)
            if n < 3:
                return 0
                
            # Calculate sample statistics
            sample_mean = sum(data) / n
            sample_variance = sum((x - sample_mean) ** 2 for x in data) / n
            sample_std = math.sqrt(sample_variance) if sample_variance > 0 else 0
            
            if sample_std == 0:
                return 0
                
            # Compute skewness coefficient
            normalized_third_moment = sum(((x - sample_mean) / sample_std) ** 3 for x in data)
            sample_skewness = (normalized_third_moment * n) / ((n - 1) * (n - 2))
            
            return sample_skewness
            
        except (ZeroDivisionError, ValueError):
            return 0
    
    @staticmethod
    def kurtosis(data):
        # Robust computation with explicit error handling
        n = len(data)
        
        try:
            # Compute basic statistics
            sample_mean = sum(data) / n
            deviations_squared = [(x - sample_mean) ** 2 for x in data]
            sample_variance = sum(deviations_squared) / n
            
            # Handle zero variance case
            if sample_variance == 0 or sample_variance < 1e-10:
                return math.nan
                
            # Calculate fourth central moment
            fourth_central_moment = sum((x - sample_mean) ** 4 for x in data) / n
            
            # Compute excess kurtosis (subtract 3 for normal distribution baseline)
            kurt_coefficient = (fourth_central_moment / (sample_variance ** 2)) - 3
            
            return kurt_coefficient
            
        except (ZeroDivisionError, OverflowError):
            return math.nan
    
    @staticmethod
    def pdf(data, mu, sigma):
        # Vectorized-style computation with pre-calculated constants
        if sigma <= 0:
            return [0] * len(data)
            
        # Pre-compute invariant terms
        normalization_constant = 1.0 / (sigma * math.sqrt(2 * math.pi))
        sigma_squared_doubled = 2 * sigma ** 2
        
        # Compute probability density for each data point
        probability_densities = []
        for x in data:
            exponent_term = -((x - mu) ** 2) / sigma_squared_doubled
            density = normalization_constant * math.exp(exponent_term)
            probability_densities.append(density)
            
        return probability_densities

import math

class DataAnalysis:

    @staticmethod
    def calculate_correlation(data1, data2):
        n = len(data1)
        avg1 = sum(data1) / n
        avg2 = sum(data2) / n

        numerator = sum((data1[i] - avg1) * (data2[i] - avg2) for i in range(n))
        denominator = math.sqrt(sum((data1[i] - avg1) ** 2 for i in range(n))) * math.sqrt(sum((data2[i] - avg2) ** 2 for i in range(n)))

        return numerator / denominator if denominator != 0 else 0
    
    @staticmethod
    def calculate_skewness(data):
        n = len(data)
        avg = sum(data) / n
        variance = sum((x - avg) ** 2 for x in data) / n
        std_dev = math.sqrt(variance)

        skewness = sum((x - avg) ** 3 for x in data) * n / ((n - 1) * (n - 2) * std_dev ** 3) if std_dev != 0 else 0

        return skewness
    
    @staticmethod
    def calculate_kurtosis(data):
        n = len(data)
        avg = sum(data) / n
        std_dev = math.sqrt(sum((x - avg) ** 2 for x in data) / n)

        if std_dev == 0:
            return math.nan

        centered_data = [(x - avg) for x in data]
        fourth_moment = sum(x ** 4 for x in centered_data) / n

        kurtosis_value = (fourth_moment / std_dev ** 4) - 3

        return kurtosis_value
    
    @staticmethod
    def pdf_values(data, mu, sigma):
        pdf_values = [1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-0.5 * ((x - mu) / sigma) ** 2) for x in data]
        return pdf_values

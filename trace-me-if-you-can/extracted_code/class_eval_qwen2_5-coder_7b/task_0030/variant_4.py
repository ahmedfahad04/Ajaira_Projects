import numpy as np

class StatisticalAnalysis:
    def __init__(self, data_points):
        self.data_points = np.array(data_points)

    def compute_sum(self):
        return np.sum(self.data_points)

    def determine_minimum(self):
        return np.min(self.data_points)

    def determine_maximum(self):
        return np.max(self.data_points)

    def calculate_variance(self):
        return round(np.var(self.data_points), 2)

    def calculate_std_dev(self):
        return round(np.std(self.data_points), 2)

    def calculate_correlation(self):
        return np.corrcoef(self.data_points, rowvar=False)

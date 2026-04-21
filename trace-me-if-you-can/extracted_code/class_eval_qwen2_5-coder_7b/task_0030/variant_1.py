import numpy as np

class DataAnalysis:
    def __init__(self, values):
        self.values = np.array(values)

    def calculate_total(self):
        return np.sum(self.values)

    def find_lowest(self):
        return np.min(self.values)

    def find_highest(self):
        return np.max(self.values)

    def calculate_variance(self):
        return round(np.var(self.values), 2)

    def calculate_standard_deviation(self):
        return round(np.std(self.values), 2)

    def compute_correlation(self):
        return np.corrcoef(self.values, rowvar=False)

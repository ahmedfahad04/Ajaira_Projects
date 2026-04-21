import numpy as np

class StatisticsCalculator:
    def __init__(self, dataset):
        self.dataset = np.array(dataset)

    def sum_values(self):
        return np.sum(self.dataset)

    def minimum_value(self):
        return np.min(self.dataset)

    def maximum_value(self):
        return np.max(self.dataset)

    def variance_value(self):
        return round(np.var(self.dataset), 2)

    def standard_deviation_value(self):
        return round(np.std(self.dataset), 2)

    def correlation_value(self):
        return np.corrcoef(self.dataset, rowvar=False)

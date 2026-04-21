import numpy as np

class DataMetrics:
    def __init__(self, numbers):
        self.numbers = np.array(numbers)

    def total(self):
        return np.sum(self.numbers)

    def smallest(self):
        return np.min(self.numbers)

    def largest(self):
        return np.max(self.numbers)

    def variance(self):
        return round(np.var(self.numbers), 2)

    def standard_dev(self):
        return round(np.std(self.numbers), 2)

    def correlation(self):
        return np.corrcoef(self.numbers, rowvar=False)

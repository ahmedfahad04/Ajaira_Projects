import numpy as np

class DataStats:
    def __init__(self, dataset):
        self.dataset = np.array(dataset)

    def get_total(self):
        return np.sum(self.dataset)

    def get_smallest(self):
        return np.min(self.dataset)

    def get_largest(self):
        return np.max(self.dataset)

    def get_variance(self):
        return round(np.var(self.dataset), 2)

    def get_std_dev(self):
        return round(np.std(self.dataset), 2)

    def get_correlation(self):
        return np.corrcoef(self.dataset, rowvar=False)

import numpy as np
from functools import partial

def create_statistics_calculator(data):
    array_data = np.array(data)
    
    operations = {
        'sum': lambda x: np.sum(x),
        'min': lambda x: np.min(x),
        'max': lambda x: np.max(x),
        'variance': lambda x: round(np.var(x), 2),
        'std_deviation': lambda x: round(np.std(x), 2),
        'correlation': lambda x: np.corrcoef(x, rowvar=False)
    }
    
    return {name: partial(func, array_data) for name, func in operations.items()}

class DataStatistics2:
    def __init__(self, data):
        self.stats = create_statistics_calculator(data)
    
    def get_sum(self):
        return self.stats['sum']()
    
    def get_min(self):
        return self.stats['min']()
    
    def get_max(self):
        return self.stats['max']()
    
    def get_variance(self):
        return self.stats['variance']()
    
    def get_std_deviation(self):
        return self.stats['std_deviation']()
    
    def get_correlation(self):
        return self.stats['correlation']()

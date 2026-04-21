import numpy as np
from abc import ABC, abstractmethod

class StatisticCalculator(ABC):
    @abstractmethod
    def calculate(self, data):
        pass

class SumCalculator(StatisticCalculator):
    def calculate(self, data):
        return np.sum(data)

class MinCalculator(StatisticCalculator):
    def calculate(self, data):
        return np.min(data)

class MaxCalculator(StatisticCalculator):
    def calculate(self, data):
        return np.max(data)

class VarianceCalculator(StatisticCalculator):
    def calculate(self, data):
        return round(np.var(data), 2)

class StdDeviationCalculator(StatisticCalculator):
    def calculate(self, data):
        return round(np.std(data), 2)

class CorrelationCalculator(StatisticCalculator):
    def calculate(self, data):
        return np.corrcoef(data, rowvar=False)

class DataStatistics2:
    def __init__(self, data):
        self.data = np.array(data)
        self._calculators = {
            'sum': SumCalculator(),
            'min': MinCalculator(),
            'max': MaxCalculator(),
            'variance': VarianceCalculator(),
            'std_deviation': StdDeviationCalculator(),
            'correlation': CorrelationCalculator()
        }
    
    def _execute_calculation(self, calc_type):
        return self._calculators[calc_type].calculate(self.data)
    
    def get_sum(self):
        return self._execute_calculation('sum')
    
    def get_min(self):
        return self._execute_calculation('min')
    
    def get_max(self):
        return self._execute_calculation('max')
    
    def get_variance(self):
        return self._execute_calculation('variance')
    
    def get_std_deviation(self):
        return self._execute_calculation('std_deviation')
    
    def get_correlation(self):
        return self._execute_calculation('correlation')

import numpy as np

class StatisticDescriptor:
    def __init__(self, func, round_digits=None):
        self.func = func
        self.round_digits = round_digits
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = f'_{name}'
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        
        if not hasattr(obj, self.name):
            result = self.func(obj.data)
            if self.round_digits is not None:
                result = round(result, self.round_digits)
            setattr(obj, self.name, result)
        
        return getattr(obj, self.name)

class DataStatistics2:
    def __init__(self, data):
        self.data = np.array(data)
    
    _sum_value = StatisticDescriptor(np.sum)
    _min_value = StatisticDescriptor(np.min)
    _max_value = StatisticDescriptor(np.max)
    _variance_value = StatisticDescriptor(np.var, 2)
    _std_deviation_value = StatisticDescriptor(np.std, 2)
    _correlation_value = StatisticDescriptor(lambda x: np.corrcoef(x, rowvar=False))
    
    def get_sum(self):
        return self._sum_value
    
    def get_min(self):
        return self._min_value
    
    def get_max(self):
        return self._max_value
    
    def get_variance(self):
        return self._variance_value
    
    def get_std_deviation(self):
        return self._std_deviation_value
    
    def get_correlation(self):
        return self._correlation_value

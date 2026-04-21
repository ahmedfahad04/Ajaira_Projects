import numpy as np

class DataStatistics2:
    def __init__(self, data):
        self._data = np.array(data)
        self._cache = {}
    
    def _compute_if_needed(self, key, computation):
        if key not in self._cache:
            self._cache[key] = computation()
        return self._cache[key]
    
    def get_sum(self):
        return self._compute_if_needed('sum', lambda: np.sum(self._data))
    
    def get_min(self):
        return self._compute_if_needed('min', lambda: np.min(self._data))
    
    def get_max(self):
        return self._compute_if_needed('max', lambda: np.max(self._data))
    
    def get_variance(self):
        return self._compute_if_needed('variance', lambda: round(np.var(self._data), 2))
    
    def get_std_deviation(self):
        return self._compute_if_needed('std_deviation', lambda: round(np.std(self._data), 2))
    
    def get_correlation(self):
        return self._compute_if_needed('correlation', lambda: np.corrcoef(self._data, rowvar=False))

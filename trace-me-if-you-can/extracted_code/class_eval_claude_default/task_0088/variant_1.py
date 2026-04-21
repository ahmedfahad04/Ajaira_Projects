import math

class TriCalculator:
    
    def __init__(self):
        pass

    def cos(self, x):
        x_rad = math.radians(x)
        result = sum((-1)**k * (x_rad**(2*k)) / math.factorial(2*k) for k in range(50))
        return round(result, 10)

    def sin(self, x):
        x_rad = math.radians(x)
        result = sum((-1)**k * (x_rad**(2*k+1)) / math.factorial(2*k+1) for k in range(50))
        return round(result, 10)

    def tan(self, x):
        cos_val = self.cos(x)
        return round(self.sin(x) / cos_val, 10) if cos_val != 0 else False

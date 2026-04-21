from functools import reduce
from operator import mul

class Interpolation:
    def __init__(self):
        pass

    @staticmethod
    def interpolate_1d(x, y, x_interp):
        def lerp(x1, y1, x2, y2, xi):
            return y1 + (y2 - y1) * (xi - x1) / (x2 - x1)
        
        interpolated_values = []
        
        for xi in x_interp:
            # Use filter and next to find the interval
            try:
                i = next(i for i in range(len(x) - 1) if x[i] <= xi <= x[i+1])
                interpolated_values.append(lerp(x[i], y[i], x[i+1], y[i+1], xi))
            except StopIteration:
                pass  # Point not in range
                
        return interpolated_values
    
    @staticmethod
    def interpolate_2d(x, y, z, x_interp, y_interp):
        def weighted_average(corners, weights):
            return sum(c * w for c, w in zip(corners, weights))
        
        result = []
        
        for xi, yi in zip(x_interp, y_interp):
            # Find intervals using generator expressions
            x_range = next((i for i in range(len(x) - 1) if x[i] <= xi <= x[i+1]), None)
            
            if x_range is not None:
                y_range = next((j for j in range(len(y) - 1) if y[j] <= yi <= y[j+1]), None)
                
                if y_range is not None:
                    i, j = x_range, y_range
                    
                    # Calculate weights for bilinear interpolation
                    dx_total = x[i+1] - x[i]
                    dy_total = y[j+1] - y[j]
                    
                    w1 = (x[i+1] - xi) * (y[j+1] - yi) / (dx_total * dy_total)
                    w2 = (xi - x[i]) * (y[j+1] - yi) / (dx_total * dy_total)
                    w3 = (x[i+1] - xi) * (yi - y[j]) / (dx_total * dy_total)
                    w4 = (xi - x[i]) * (yi - y[j]) / (dx_total * dy_total)
                    
                    corners = [z[i][j], z[i+1][j], z[i][j+1], z[i+1][j+1]]
                    weights = [w1, w2, w3, w4]
                    
                    result.append(weighted_average(corners, weights))
        
        return result

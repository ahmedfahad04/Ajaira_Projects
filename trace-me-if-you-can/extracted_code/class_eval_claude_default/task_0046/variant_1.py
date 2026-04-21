import bisect

class Interpolation:
    def __init__(self):
        pass

    @staticmethod
    def interpolate_1d(x, y, x_interp):
        def find_interval_and_interpolate(xi):
            # Use binary search to find the interval
            idx = bisect.bisect_left(x, xi)
            if idx == 0:
                idx = 1
            elif idx >= len(x):
                idx = len(x) - 1
            
            i = idx - 1
            return y[i] + (y[i+1] - y[i]) * (xi - x[i]) / (x[i+1] - x[i])
        
        return [find_interval_and_interpolate(xi) for xi in x_interp]
    
    @staticmethod
    def interpolate_2d(x, y, z, x_interp, y_interp):
        def bilinear_interpolate(xi, yi):
            # Find x interval
            x_idx = bisect.bisect_left(x, xi)
            if x_idx == 0:
                x_idx = 1
            elif x_idx >= len(x):
                x_idx = len(x) - 1
            i = x_idx - 1
            
            # Find y interval
            y_idx = bisect.bisect_left(y, yi)
            if y_idx == 0:
                y_idx = 1
            elif y_idx >= len(y):
                y_idx = len(y) - 1
            j = y_idx - 1
            
            # Bilinear interpolation
            dx = x[i+1] - x[i]
            dy = y[j+1] - y[j]
            return ((z[i][j] * (x[i+1] - xi) * (y[j+1] - yi) +
                     z[i+1][j] * (xi - x[i]) * (y[j+1] - yi) +
                     z[i][j+1] * (x[i+1] - xi) * (yi - y[j]) +
                     z[i+1][j+1] * (xi - x[i]) * (yi - y[j])) / (dx * dy))
        
        return [bilinear_interpolate(xi, yi) for xi, yi in zip(x_interp, y_interp)]

class Interpolation:
    def __init__(self):
        pass

    @staticmethod
    def interpolate_1d(x, y, x_interp):
        result = []
        x_array = list(x)
        y_array = list(y)
        
        for xi in x_interp:
            # Find the appropriate interval using enumerate
            interval_found = False
            for idx, (x_left, x_right) in enumerate(zip(x_array[:-1], x_array[1:])):
                if x_left <= xi <= x_right:
                    # Linear interpolation formula
                    t = (xi - x_left) / (x_right - x_left)
                    yi = y_array[idx] * (1 - t) + y_array[idx + 1] * t
                    result.append(yi)
                    interval_found = True
                    break
        return result
    
    @staticmethod
    def interpolate_2d(x, y, z, x_interp, y_interp):
        result = []
        
        for xi, yi in zip(x_interp, y_interp):
            # Find x and y intervals using nested comprehensions
            x_intervals = [(i, x[i], x[i+1]) for i in range(len(x)-1) if x[i] <= xi <= x[i+1]]
            
            if x_intervals:
                i, x_left, x_right = x_intervals[0]
                y_intervals = [(j, y[j], y[j+1]) for j in range(len(y)-1) if y[j] <= yi <= y[j+1]]
                
                if y_intervals:
                    j, y_bottom, y_top = y_intervals[0]
                    
                    # Normalized coordinates
                    tx = (xi - x_left) / (x_right - x_left)
                    ty = (yi - y_bottom) / (y_top - y_bottom)
                    
                    # Bilinear interpolation using normalized coordinates
                    zi = (z[i][j] * (1-tx) * (1-ty) +
                          z[i+1][j] * tx * (1-ty) +
                          z[i][j+1] * (1-tx) * ty +
                          z[i+1][j+1] * tx * ty)
                    result.append(zi)
        
        return result

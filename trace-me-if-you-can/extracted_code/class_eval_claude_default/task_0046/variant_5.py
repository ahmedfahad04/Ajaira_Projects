class Interpolation:
    def __init__(self):
        pass

    @staticmethod
    def interpolate_1d(x, y, x_interp):
        # Create a mapping approach using dictionary for interval lookup
        intervals = {}
        for i in range(len(x) - 1):
            for xi in x_interp:
                if x[i] <= xi <= x[i+1] and xi not in intervals:
                    intervals[xi] = i
        
        # Process interpolation using the pre-computed intervals
        y_interp = []
        for xi in x_interp:
            if xi in intervals:
                i = intervals[xi]
                # Use parametric form: P(t) = P0 + t(P1 - P0) where t = (xi - x[i])/(x[i+1] - x[i])
                t = (xi - x[i]) / (x[i+1] - x[i])
                yi = y[i] + t * (y[i+1] - y[i])
                y_interp.append(yi)
        
        return y_interp
    
    @staticmethod
    def interpolate_2d(x, y, z, x_interp, y_interp):
        # Recursive approach: interpolate in x-direction first, then y-direction
        def interpolate_along_x(z_row, xi, x_start_idx):
            i = x_start_idx
            t = (xi - x[i]) / (x[i+1] - x[i])
            return z_row[i] + t * (z_row[i+1] - z_row[i])
        
        def interpolate_along_y(z_col, yi, y_start_idx):
            j = y_start_idx
            t = (yi - y[j]) / (y[j+1] - y[j])
            return z_col[j] + t * (z_col[j+1] - z_col[j])
        
        z_interp = []
        
        for xi, yi in zip(x_interp, y_interp):
            # Find intervals
            x_idx = next((i for i in range(len(x) - 1) if x[i] <= xi <= x[i+1]), None)
            y_idx = next((j for j in range(len(y) - 1) if y[j] <= yi <= y[j+1]), None)
            
            if x_idx is not None and y_idx is not None:
                # First interpolate along x at both y boundaries
                z_at_y0 = interpolate_along_x([z[x_idx][y_idx], z[x_idx+1][y_idx]], xi, x_idx)
                z_at_y1 = interpolate_along_x([z[x_idx][y_idx+1], z[x_idx+1][y_idx+1]], xi, x_idx)
                
                # Then interpolate along y
                zi = interpolate_along_y([z_at_y0, z_at_y1], yi, y_idx)
                z_interp.append(zi)
        
        return z_interp

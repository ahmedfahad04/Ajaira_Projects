import numpy as np

class Interpolation:
    def __init__(self):
        pass

    @staticmethod
    def interpolate_1d(x, y, x_interp):
        # Convert to numpy arrays for vectorized operations
        x_arr = np.array(x)
        y_arr = np.array(y)
        x_interp_arr = np.array(x_interp)
        
        result = []
        
        for xi in x_interp_arr:
            # Use numpy's searchsorted for efficient interval finding
            idx = np.searchsorted(x_arr, xi, side='right') - 1
            idx = max(0, min(idx, len(x_arr) - 2))  # Clamp to valid range
            
            # Vectorized linear interpolation
            slope = (y_arr[idx + 1] - y_arr[idx]) / (x_arr[idx + 1] - x_arr[idx])
            yi = y_arr[idx] + slope * (xi - x_arr[idx])
            result.append(float(yi))
            
        return result
    
    @staticmethod
    def interpolate_2d(x, y, z, x_interp, y_interp):
        x_arr = np.array(x)
        y_arr = np.array(y)
        z_arr = np.array(z)
        
        result = []
        
        for xi, yi in zip(x_interp, y_interp):
            # Find intervals using numpy
            i = np.searchsorted(x_arr, xi, side='right') - 1
            j = np.searchsorted(y_arr, yi, side='right') - 1
            
            # Clamp indices
            i = max(0, min(i, len(x_arr) - 2))
            j = max(0, min(j, len(y_arr) - 2))
            
            # Extract corner values
            corners = np.array([[z_arr[i, j], z_arr[i, j+1]], 
                               [z_arr[i+1, j], z_arr[i+1, j+1]]])
            
            # Normalized coordinates
            tx = (xi - x_arr[i]) / (x_arr[i+1] - x_arr[i])
            ty = (yi - y_arr[j]) / (y_arr[j+1] - y_arr[j])
            
            # Bilinear interpolation using matrix operations
            x_weights = np.array([1-tx, tx])
            y_weights = np.array([1-ty, ty])
            
            zi = np.dot(x_weights, np.dot(corners, y_weights))
            result.append(float(zi))
            
        return result

class CurveFitting:
    @staticmethod
    def interpolate_1d(x_points, y_points, x_to_interpolate):
        y_interpolated = []
        for x_val in x_to_interpolate:
            for i in range(len(x_points) - 1):
                if x_points[i] <= x_val <= x_points[i + 1]:
                    yi = y_points[i] + (y_points[i + 1] - y_points[i]) * (x_val - x_points[i]) / (x_points[i + 1] - x_points[i])
                    y_interpolated.append(yi)
                    break
        return y_interpolated
    
    @staticmethod
    def interpolate_2d(x_points, y_points, z_values, x_to_interpolate, y_to_interpolate):
        z_interpolated = []
        for x_val, y_val in zip(x_to_interpolate, y_to_interpolate):
            for i in range(len(x_points) - 1):
                if x_points[i] <= x_val <= x_points[i + 1]:
                    for j in range(len(y_points) - 1):
                        if y_points[j] <= y_val <= y_points[j + 1]:
                            z00 = z_values[i][j]
                            z01 = z_values[i][j + 1]
                            z10 = z_values[i + 1][j]
                            z11 = z_values[i + 1][j + 1]
                            zi = (z00 * (x_points[i + 1] - x_val) * (y_points[j + 1] - y_val) +
                                  z10 * (x_val - x_points[i]) * (y_points[j + 1] - y_val) +
                                  z01 * (x_points[i + 1] - x_val) * (y_val - y_points[j]) +
                                  z11 * (x_val - x_points[i]) * (y_val - y_points[j])) / ((x_points[i + 1] - x_points[i]) * (y_points[j + 1] - y_points[j]))
                            z_interpolated.append(zi)
                            break
                    break
        return z_interpolated

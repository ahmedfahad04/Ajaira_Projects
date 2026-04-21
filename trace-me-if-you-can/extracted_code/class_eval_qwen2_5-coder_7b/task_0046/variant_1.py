class DataInterpolation:
    @staticmethod
    def linear_1d(x_values, y_values, x_interpolate):
        interpolated_values = []
        for x_i in x_interpolate:
            for i in range(len(x_values) - 1):
                if x_values[i] <= x_i <= x_values[i + 1]:
                    yi = y_values[i] + (y_values[i + 1] - y_values[i]) * (x_i - x_values[i]) / (x_values[i + 1] - x_values[i])
                    interpolated_values.append(yi)
                    break
        return interpolated_values
    
    @staticmethod
    def bilinear_2d(x_values, y_values, z_values, x_interpolate, y_interpolate):
        interpolated_values = []
        for x_i, y_i in zip(x_interpolate, y_interpolate):
            for i in range(len(x_values) - 1):
                if x_values[i] <= x_i <= x_values[i + 1]:
                    for j in range(len(y_values) - 1):
                        if y_values[j] <= y_i <= y_values[j + 1]:
                            z00 = z_values[i][j]
                            z01 = z_values[i][j + 1]
                            z10 = z_values[i + 1][j]
                            z11 = z_values[i + 1][j + 1]
                            zi = (z00 * (x_values[i + 1] - x_i) * (y_values[j + 1] - y_i) +
                                  z10 * (x_i - x_values[i]) * (y_values[j + 1] - y_i) +
                                  z01 * (x_values[i + 1] - x_i) * (y_i - y_values[j]) +
                                  z11 * (x_i - x_values[i]) * (y_i - y_values[j])) / ((x_values[i + 1] - x_values[i]) * (y_values[j + 1] - y_values[j]))
                            interpolated_values.append(zi)
                            break
                    break
        return interpolated_values

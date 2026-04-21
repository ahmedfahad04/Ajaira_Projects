class FunctionApproximation:
    @staticmethod
    def linear_interpolation(x_data, y_data, x_points):
        y_interpolated = []
        for x_point in x_points:
            for i in range(len(x_data) - 1):
                if x_data[i] <= x_point <= x_data[i + 1]:
                    yi = y_data[i] + (y_data[i + 1] - y_data[i]) * (x_point - x_data[i]) / (x_data[i + 1] - x_data[i])
                    y_interpolated.append(yi)
                    break
        return y_interpolated
    
    @staticmethod
    def bicubic_interpolation(x_data, y_data, z_data, x_points, y_points):
        z_interpolated = []
        for x_point, y_point in zip(x_points, y_points):
            for i in range(len(x_data) - 1):
                if x_data[i] <= x_point <= x_data[i + 1]:
                    for j in range(len(y_data) - 1):
                        if y_data[j] <= y_point <= y_data[j + 1]:
                            z00 = z_data[i][j]
                            z01 = z_data[i][j + 1]
                            z10 = z_data[i + 1][j]
                            z11 = z_data[i + 1][j + 1]
                            zi = (z00 * (x_data[i + 1] - x_point) * (y_data[j + 1] - y_point) +
                                  z10 * (x_point - x_data[i]) * (y_data[j + 1] - y_point) +
                                  z01 * (x_data[i + 1] - x_point) * (y_point - y_data[j]) +
                                  z11 * (x_point - x_data[i]) * (y_point - y_data[j])) / ((x_data[i + 1] - x_data[i]) * (y_data[j + 1] - y_data[j]))
                            z_interpolated.append(zi)
                            break
                    break
        return z_interpolated

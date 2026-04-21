class MathematicalInterpolation:
    @staticmethod
    def perform_1d_interpolation(x_coords, y_coords, x_values_to_interpolate):
        interpolated_y_values = []
        for x_val in x_values_to_interpolate:
            for i in range(len(x_coords) - 1):
                if x_coords[i] <= x_val <= x_coords[i + 1]:
                    yi = y_coords[i] + (y_coords[i + 1] - y_coords[i]) * (x_val - x_coords[i]) / (x_coords[i + 1] - x_coords[i])
                    interpolated_y_values.append(yi)
                    break
        return interpolated_y_values
    
    @staticmethod
    def perform_2d_interpolation(x_coords, y_coords, z_matrix, x_values_to_interpolate, y_values_to_interpolate):
        interpolated_z_values = []
        for x_val, y_val in zip(x_values_to_interpolate, y_values_to_interpolate):
            for i in range(len(x_coords) - 1):
                if x_coords[i] <= x_val <= x_coords[i + 1]:
                    for j in range(len(y_coords) - 1):
                        if y_coords[j] <= y_val <= y_coords[j + 1]:
                            z00 = z_matrix[i][j]
                            z01 = z_matrix[i][j + 1]
                            z10 = z_matrix[i + 1][j]
                            z11 = z_matrix[i + 1][j + 1]
                            zi = (z00 * (x_coords[i + 1] - x_val) * (y_coords[j + 1] - y_val) +
                                  z10 * (x_val - x_coords[i]) * (y_coords[j + 1] - y_val) +
                                  z01 * (x_coords[i + 1] - x_val) * (y_val - y_coords[j]) +
                                  z11 * (x_val - x_coords[i]) * (y_val - y_coords[j])) / ((x_coords[i + 1] - x_coords[i]) * (y_coords[j + 1] - y_coords[j]))
                            interpolated_z_values.append(zi)
                            break
                    break
        return interpolated_z_values

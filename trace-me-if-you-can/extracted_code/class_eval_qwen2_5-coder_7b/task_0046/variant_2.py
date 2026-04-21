class NumericInterpolation:
    @staticmethod
    def one_d_interpolation(x, y, x_new):
        result = []
        for x_point in x_new:
            for i in range(len(x) - 1):
                if x[i] <= x_point <= x[i + 1]:
                    yi = y[i] + (y[i + 1] - y[i]) * (x_point - x[i]) / (x[i + 1] - x[i])
                    result.append(yi)
                    break
        return result
    
    @staticmethod
    def two_d_interpolation(x, y, z, x_new, y_new):
        result = []
        for x_point, y_point in zip(x_new, y_new):
            for i in range(len(x) - 1):
                if x[i] <= x_point <= x[i + 1]:
                    for j in range(len(y) - 1):
                        if y[j] <= y_point <= y[j + 1]:
                            z00 = z[i][j]
                            z01 = z[i][j + 1]
                            z10 = z[i + 1][j]
                            z11 = z[i + 1][j + 1]
                            zi = (z00 * (x[i + 1] - x_point) * (y[j + 1] - y_point) +
                                  z10 * (x_point - x[i]) * (y[j + 1] - y_point) +
                                  z01 * (x[i + 1] - x_point) * (y_point - y[j]) +
                                  z11 * (x_point - x[i]) * (y_point - y[j])) / ((x[i + 1] - x[i]) * (y[j + 1] - y[j]))
                            result.append(zi)
                            break
                    break
        return result

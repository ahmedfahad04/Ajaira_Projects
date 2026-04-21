import statistics

class DataAnalysis:
    @staticmethod
    def find_median(values):
        sorted_values = sorted(values)
        length = len(sorted_values)
        if length % 2 == 1:
            return sorted_values[length // 2]
        else:
            return (sorted_values[length // 2 - 1] + sorted_values[length // 2]) / 2

    @staticmethod
    def find_mode(values):
        count_dict = {}
        for value in values:
            count_dict[value] = count_dict.get(value, 0) + 1
        max_freq = max(count_dict.values())
        mode_list = [value for value, freq in count_dict.items() if freq == max_freq]
        return mode_list

    @staticmethod
    def calculate_correlation(x_values, y_values):
        size = len(x_values)
        avg_x = sum(x_values) / size
        avg_y = sum(y_values) / size
        numerator = sum((x_values[i] - avg_x) * (y_values[i] - avg_y) for i in range(size))
        denominator = math.sqrt(sum((x_values[i] - avg_x) ** 2 for i in range(size)) * sum((y_values[i] - avg_y) ** 2 for i in range(size)))
        if denominator == 0:
            return None
        return numerator / denominator

    @staticmethod
    def calculate_mean(values):
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def create_correlation_matrix(dataset):
        matrix = []
        for col_index in range(len(dataset[0])):
            column_data = [dataset[row][col_index] for row in range(len(dataset))]
            row_data = []
            for another_col_index in range(len(dataset[0])):
                another_column_data = [dataset[row][another_col_index] for row in range(len(dataset))]
                correlation = DataAnalysis.calculate_correlation(column_data, another_column_data)
                row_data.append(correlation)
            matrix.append(row_data)
        return matrix

    @staticmethod
    def compute_standard_deviation(values):
        if len(values) < 2:
            return None
        avg_value = DataAnalysis.calculate_mean(values)
        variance = sum((value - avg_value) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(variance)

    @staticmethod
    def compute_z_scores(values):
        avg = DataAnalysis.calculate_mean(values)
        std_dev = DataAnalysis.compute_standard_deviation(values)
        if std_dev is None or std_dev == 0:
            return None
        return [(value - avg) / std_dev for value in values]

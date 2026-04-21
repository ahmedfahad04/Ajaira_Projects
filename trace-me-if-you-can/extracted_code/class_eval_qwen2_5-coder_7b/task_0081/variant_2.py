import statistics

class StatisticalFunctions:
    @staticmethod
    def calculate_median(data):
        sorted_data = sorted(data)
        num_elements = len(sorted_data)
        if num_elements % 2 == 1:
            return sorted_data[num_elements // 2]
        else:
            return (sorted_data[num_elements // 2 - 1] + sorted_data[num_elements // 2]) / 2

    @staticmethod
    def calculate_mode(data):
        count_dict = {}
        for value in data:
            count_dict[value] = count_dict.get(value, 0) + 1
        highest_frequency = max(count_dict.values())
        mode_values = [value for value, frequency in count_dict.items() if frequency == highest_frequency]
        return mode_values

    @staticmethod
    def calculate_correlation(x, y):
        data_size = len(x)
        avg_x = sum(x) / data_size
        avg_y = sum(y) / data_size
        numerator = sum((x[i] - avg_x) * (y[i] - avg_y) for i in range(data_size))
        denominator = math.sqrt(sum((x[i] - avg_x) ** 2 for i in range(data_size)) * sum((y[i] - avg_y) ** 2 for i in range(data_size)))
        if denominator == 0:
            return None
        return numerator / denominator

    @staticmethod
    def calculate_mean(data):
        if len(data) == 0:
            return None
        return sum(data) / len(data)

    @staticmethod
    def create_correlation_matrix(data):
        matrix = []
        for i in range(len(data[0])):
            column1 = [row[i] for row in data]
            row_data = []
            for j in range(len(data[0])):
                column2 = [row[j] for row in data]
                correlation = StatisticalFunctions.calculate_correlation(column1, column2)
                row_data.append(correlation)
            matrix.append(row_data)
        return matrix

    @staticmethod
    def compute_standard_deviation(data):
        if len(data) < 2:
            return None
        average_value = StatisticalFunctions.calculate_mean(data)
        variance = sum((value - average_value) ** 2 for value in data) / (len(data) - 1)
        return math.sqrt(variance)

    @staticmethod
    def compute_z_scores(data):
        avg = StatisticalFunctions.calculate_mean(data)
        std_dev = StatisticalFunctions.compute_standard_deviation(data)
        if std_dev is None or std_dev == 0:
            return None
        return [(value - avg) / std_dev for value in data]

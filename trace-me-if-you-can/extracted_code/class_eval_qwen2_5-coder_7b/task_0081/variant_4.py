import statistics

class AnalyzeData:
    @staticmethod
    def find_median(data):
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 1:
            return sorted_data[n // 2]
        else:
            return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2

    @staticmethod
    def find_mode(data):
        count_dict = {}
        for value in data:
            count_dict[value] = count_dict.get(value, 0) + 1
        max_count = max(count_dict.values())
        mode_values = [value for value, count in count_dict.items() if count == max_count]
        return mode_values

    @staticmethod
    def calculate_correlation(x, y):
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y))
        if denominator == 0:
            return None
        return numerator / denominator

    @staticmethod
    def calculate_mean(data):
        if not data:
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
                correlation = AnalyzeData.calculate_correlation(column1, column2)
                row_data.append(correlation)
            matrix.append(row_data)
        return matrix

    @staticmethod
    def compute_standard_deviation(data):
        if len(data) < 2:
            return None
        avg_value = AnalyzeData.calculate_mean(data)
        variance = sum((value - avg_value) ** 2 for value in data) / (len(data) - 1)
        return math.sqrt(variance)

    @staticmethod
    def compute_z_scores(data):
        avg = AnalyzeData.calculate_mean(data)
        std_dev = AnalyzeData.compute_standard_deviation(data)
        if std_dev is None or std_dev == 0:
            return None
        return [(value - avg) / std_dev for value in data]

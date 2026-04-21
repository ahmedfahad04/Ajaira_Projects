import statistics

class Stats:
    @staticmethod
    def med(data):
        sorted_data = sorted(data)
        count = len(sorted_data)
        if count % 2 == 1:
            return sorted_data[count // 2]
        else:
            return (sorted_data[count // 2 - 1] + sorted_data[count // 2]) / 2

    @staticmethod
    def mode(data):
        count_dict = {}
        for value in data:
            count_dict[value] = count_dict.get(value, 0) + 1
        max_count = max(count_dict.values())
        mode_values = [value for value, count in count_dict.items() if count == max_count]
        return mode_values

    @staticmethod
    def correlation(x, y):
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y))
        if denominator == 0:
            return None
        return numerator / denominator

    @staticmethod
    def mean(data):
        if len(data) == 0:
            return None
        return sum(data) / len(data)

    @staticmethod
    def correlation_matrix(data):
        matrix = []
        for i in range(len(data[0])):
            column1 = [row[i] for row in data]
            row_data = []
            for j in range(len(data[0])):
                column2 = [row[j] for row in data]
                correlation = Stats.correlation(column1, column2)
                row_data.append(correlation)
            matrix.append(row_data)
        return matrix

    @staticmethod
    def std_dev(data):
        n = len(data)
        if n < 2:
            return None
        mean_value = Stats.mean(data)
        variance = sum((x - mean_value) ** 2 for x in data) / (n - 1)
        return math.sqrt(variance)

    @staticmethod
    def z_scores(data):
        mean = Stats.mean(data)
        std_deviation = Stats.std_dev(data)
        if std_deviation is None or std_deviation == 0:
            return None
        return [(x - mean) / std_deviation for x in data]

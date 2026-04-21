import numpy as np

class CohenKappa:

    @staticmethod
    def calculate_kappa(test_data, category_count):
        data = np.matrix(test_data)
        observed_agreement = 0.0
        for i in range(category_count):
            observed_agreement += data[i, i]
        row_totals = np.sum(data, axis=1)
        col_totals = np.sum(data, axis=0)
        total_sum = np.sum(data)
        expected_agreement = (row_totals * col_totals) / total_sum / total_sum
        normalized_agreement = observed_agreement / total_sum
        cohens_coefficient = (normalized_agreement - expected_agreement) / (1 - expected_agreement)
        return cohens_coefficient

    @staticmethod
    def fleiss_kappa(data, sample_size, category_count, response_count):
        matrix = np.matrix(data, dtype=float)
        ones_vector = np.ones((category_count, 1))
        total_sum = 0.0
        self_agreement = 0.0
        for i in range(sample_size):
            temp_total = 0.0
            for j in range(category_count):
                total_sum += matrix[i, j]
                temp_total += matrix[i, j] ** 2
            temp_total -= response_count
            temp_total /= (response_count - 1) * response_count
            self_agreement += temp_total
        average_self_agreement = self_agreement / sample_size
        expected_self_agreement = np.sum((np.sum(matrix, axis=0) / total_sum) ** 2) * ones_vector
        kappa_value = (average_self_agreement - expected_self_agreement) / (1 - expected_self_agreement)
        return kappa_value[0, 0]

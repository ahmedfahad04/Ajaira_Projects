import numpy as np

class AgreementEvaluator:

    @staticmethod
    def calculate_kappa(test_data, category_count):
        matrix = np.matrix(test_data)
        self_agreement = 0.0
        for i in range(category_count):
            self_agreement += matrix[i, i]
        row_sums = np.sum(matrix, axis=1)
        col_sums = np.sum(matrix, axis=0)
        total_sum = np.sum(matrix)
        expected_self_agreement = (row_sums * col_sums) / total_sum / total_sum
        normalized_self_agreement = self_agreement / total_sum
        cohens_coefficient = (normalized_self_agreement - expected_self_agreement) / (1 - expected_self_agreement)
        return cohens_coefficient

    @staticmethod
    def fleiss_kappa(matrix_data, sample_size, category_count, response_count):
        matrix = np.matrix(matrix_data, dtype=float)
        ones_vector = np.ones((category_count, 1))
        total_sum = 0.0
        self_agreement = 0.0
        for i in range(sample_size):
            temp_sum = 0.0
            for j in range(category_count):
                total_sum += matrix[i, j]
                temp_sum += matrix[i, j] ** 2
            temp_sum -= response_count
            temp_sum /= (response_count - 1) * response_count
            self_agreement += temp_sum
        average_self_agreement = self_agreement / sample_size
        expected_self_agreement = np.sum((np.sum(matrix, axis=0) / total_sum) ** 2) * ones_vector
        kappa_value = (average_self_agreement - expected_self_agreement) / (1 - expected_self_agreement)
        return kappa_value[0, 0]

import numpy as np


class KappaCalculator:

    @staticmethod
    def kappa(testData, k):
        dataMat = np.array(testData)
        diagonal_sum = np.trace(dataMat[:k, :k])
        total_sum = np.sum(dataMat)
        
        row_marginals = np.sum(dataMat, axis=1)
        col_marginals = np.sum(dataMat, axis=0)
        
        P0 = diagonal_sum / total_sum
        Pe = np.dot(row_marginals, col_marginals) / (total_sum ** 2)
        
        return (P0 - Pe) / (1 - Pe)

    @staticmethod
    def fleiss_kappa(testData, N, k, n):
        dataMat = np.array(testData, dtype=float)
        
        # Calculate P0 using vectorized operations
        squared_ratings = dataMat ** 2
        row_sums_squared = np.sum(squared_ratings, axis=1)
        P0 = np.mean((row_sums_squared - n) / (n * (n - 1)))
        
        # Calculate Pe
        total_ratings = np.sum(dataMat)
        category_proportions = np.sum(dataMat, axis=0) / total_ratings
        Pe = np.sum(category_proportions ** 2)
        
        return (P0 - Pe) / (1 - Pe)

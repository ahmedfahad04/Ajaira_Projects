import numpy as np


class KappaCalculator:

    @classmethod
    def _calculate_observed_agreement(cls, confusion_matrix, k):
        """Helper method to calculate observed agreement"""
        return sum(confusion_matrix[i, i] for i in range(k))

    @classmethod
    def _calculate_expected_agreement(cls, confusion_matrix):
        """Helper method to calculate expected agreement by chance"""
        row_totals = np.sum(confusion_matrix, axis=1)
        col_totals = np.sum(confusion_matrix, axis=0)
        total = np.sum(confusion_matrix)
        return float(np.dot(row_totals, col_totals)) / (total ** 2)

    @classmethod
    def kappa(cls, testData, k):
        dataMat = np.mat(testData)
        total = float(np.sum(dataMat))
        
        observed_agreement = cls._calculate_observed_agreement(dataMat, k) / total
        expected_agreement = cls._calculate_expected_agreement(dataMat)
        
        return (observed_agreement - expected_agreement) / (1 - expected_agreement)

    @classmethod
    def fleiss_kappa(cls, testData, N, k, n):
        ratings_matrix = np.array(testData, dtype=float)
        
        # Calculate observed agreement per subject, then average
        def subject_agreement(subject_ratings):
            return (np.sum(subject_ratings ** 2) - n) / (n * (n - 1))
        
        P0 = np.mean([subject_agreement(ratings_matrix[i, :]) for i in range(N)])
        
        # Expected agreement calculation
        total_assignments = np.sum(ratings_matrix)
        marginal_proportions = np.sum(ratings_matrix, axis=0) / total_assignments
        Pe = np.sum(marginal_proportions ** 2)
        
        return (P0 - Pe) / (1 - Pe)

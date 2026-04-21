import numpy as np


class KappaCalculator:

    @staticmethod
    def kappa(testData, k):
        # Using advanced NumPy indexing and broadcasting
        confusion_matrix = np.asarray(testData)
        
        # Extract diagonal using advanced indexing
        diagonal_indices = np.arange(k)
        observed_correct = np.sum(confusion_matrix[diagonal_indices, diagonal_indices])
        
        # Broadcasting approach for marginal calculations
        total_samples = confusion_matrix.sum()
        row_marginals = confusion_matrix.sum(axis=1, keepdims=True)
        col_marginals = confusion_matrix.sum(axis=0, keepdims=True)
        
        # Use broadcasting for expected calculation
        expected_correct = np.sum(row_marginals * col_marginals) / total_samples
        
        # Probability calculations
        p_observed = observed_correct / total_samples
        p_expected = expected_correct / total_samples
        
        return (p_observed - p_expected) / (1.0 - p_expected)

    @staticmethod
    def fleiss_kappa(testData, N, k, n):
        # Matrix operations with einsum for efficiency
        rating_matrix = np.asarray(testData, dtype=np.float64)
        
        # Use einsum for efficient squared sum calculation
        squared_sums_per_subject = np.einsum('ij,ij->i', rating_matrix, rating_matrix)
        subject_agreements = (squared_sums_per_subject - n) / (n * (n - 1))
        observed_agreement = np.mean(subject_agreements)
        
        # Efficient marginal proportion calculation using einsum
        total_ratings = np.sum(rating_matrix)
        marginal_props = np.sum(rating_matrix, axis=0) / total_ratings
        expected_agreement = np.einsum('i,i->', marginal_props, marginal_props)
        
        return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)

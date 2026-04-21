import numpy as np


class KappaCalculator:

    @staticmethod
    def kappa(testData, k):
        # Dictionary-based approach for cleaner data handling
        matrix_data = {'mat': np.mat(testData), 'size': k}
        totals = {'sum': np.sum(matrix_data['mat'])}
        
        # Calculate agreements using dictionary structure
        agreements = {
            'observed': sum(matrix_data['mat'][i, i] for i in range(matrix_data['size'])),
            'marginal_row': np.sum(matrix_data['mat'], axis=1),
            'marginal_col': np.sum(matrix_data['mat'], axis=0)
        }
        
        # Normalize probabilities
        probs = {
            'observed': float(agreements['observed']) / totals['sum'],
            'expected': float(agreements['marginal_row'] * agreements['marginal_col']) / (totals['sum'] ** 2)
        }
        
        return (probs['observed'] - probs['expected']) / (1 - probs['expected'])

    @staticmethod
    def fleiss_kappa(testData, N, k, n):
        # Structured approach with clear separation of concerns
        data_structure = {
            'matrix': np.mat(testData, float),
            'dimensions': {'subjects': N, 'categories': k, 'raters': n}
        }
        
        # Calculate subject-wise agreement scores
        subject_scores = []
        for subject_idx in range(data_structure['dimensions']['subjects']):
            subject_ratings = data_structure['matrix'][subject_idx, :]
            score = (np.sum(subject_ratings ** 2) - data_structure['dimensions']['raters'])
            score /= (data_structure['dimensions']['raters'] * (data_structure['dimensions']['raters'] - 1))
            subject_scores.append(score)
        
        observed_agreement = np.mean(subject_scores)
        
        # Expected agreement calculation
        total_observations = float(np.sum(data_structure['matrix']))
        category_frequencies = np.sum(data_structure['matrix'], axis=0) / total_observations
        expected_agreement = float(np.sum(category_frequencies ** 2))
        
        return (observed_agreement - expected_agreement) / (1 - expected_agreement)

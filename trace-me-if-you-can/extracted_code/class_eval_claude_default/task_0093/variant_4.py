import numpy as np
from gensim import matutils
from numpy import dot, array


class VectorUtil:
    @staticmethod
    def similarity(vector_1, vector_2):
        # Manual normalization with explicit L2 norm computation
        norm1 = np.sqrt(np.sum(vector_1 ** 2))
        norm2 = np.sqrt(np.sum(vector_2 ** 2))
        return np.sum(vector_1 * vector_2) / (norm1 * norm2)

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)
        dot_products = dot(vectors_all, vector_1)
        similarities = dot_products / (norm * all_norms)
        return similarities

    @staticmethod
    def n_similarity(vector_list_1, vector_list_2):
        if not (len(vector_list_1) and len(vector_list_2)):
            raise ZeroDivisionError('At least one of the passed list is empty.')

        # Compute means using explicit summation
        sum1 = np.sum(array(vector_list_1), axis=0)
        sum2 = np.sum(array(vector_list_2), axis=0)
        mean1 = sum1 / len(vector_list_1)
        mean2 = sum2 / len(vector_list_2)
        
        return dot(matutils.unitvec(mean1), matutils.unitvec(mean2))

    @staticmethod
    def compute_idf_weight_dict(total_num, number_dict):
        # List comprehension with enumerate pattern
        keys = list(number_dict.keys())
        counts = [number_dict[key] for key in keys]
        
        weights = [np.log((total_num + 1) / (count + 1)) for count in counts]
        
        return {keys[i]: weights[i] for i in range(len(keys))}

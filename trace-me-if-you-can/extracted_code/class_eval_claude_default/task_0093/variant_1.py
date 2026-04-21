import numpy as np
from gensim import matutils
from numpy import dot, array


class VectorUtil:
    @staticmethod
    def similarity(vector_1, vector_2):
        # Use sklearn-style normalization approach
        v1_norm = vector_1 / np.linalg.norm(vector_1)
        v2_norm = vector_2 / np.linalg.norm(vector_2)
        return np.dot(v1_norm, v2_norm)

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        # Vectorized computation using broadcasting
        normalized_v1 = vector_1 / np.linalg.norm(vector_1)
        normalized_vectors = vectors_all / np.linalg.norm(vectors_all, axis=1, keepdims=True)
        return np.dot(normalized_vectors, normalized_v1)

    @staticmethod
    def n_similarity(vector_list_1, vector_list_2):
        if not (len(vector_list_1) and len(vector_list_2)):
            raise ZeroDivisionError('At least one of the passed list is empty.')

        mean_v1 = np.mean(array(vector_list_1), axis=0)
        mean_v2 = np.mean(array(vector_list_2), axis=0)
        
        return dot(matutils.unitvec(mean_v1), matutils.unitvec(mean_v2))

    @staticmethod
    def compute_idf_weight_dict(total_num, number_dict):
        # Direct dictionary comprehension approach
        counts = np.array(list(number_dict.values()))
        weights = np.log((total_num + 1) / (counts + 1))
        
        return {key: weight for key, weight in zip(number_dict.keys(), weights)}

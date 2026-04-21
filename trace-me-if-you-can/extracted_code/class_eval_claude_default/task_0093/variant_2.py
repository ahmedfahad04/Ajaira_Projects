import numpy as np
from gensim import matutils
from numpy import dot, array


class VectorUtil:
    @staticmethod
    def similarity(vector_1, vector_2):
        return dot(matutils.unitvec(vector_1), matutils.unitvec(vector_2))

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        # Use einsum for explicit tensor contraction
        v1_normalized = vector_1 / np.linalg.norm(vector_1)
        vectors_normalized = vectors_all / np.linalg.norm(vectors_all, axis=1)[:, np.newaxis]
        return np.einsum('ij,j->i', vectors_normalized, v1_normalized)

    @staticmethod
    def n_similarity(vector_list_1, vector_list_2):
        if not (len(vector_list_1) and len(vector_list_2)):
            raise ZeroDivisionError('At least one of the passed list is empty.')

        # Stack and compute means in one operation
        vectors_1 = np.stack(vector_list_1)
        vectors_2 = np.stack(vector_list_2)
        
        centroid_1 = vectors_1.mean(axis=0)
        centroid_2 = vectors_2.mean(axis=0)
        
        return dot(matutils.unitvec(centroid_1), matutils.unitvec(centroid_2))

    @staticmethod
    def compute_idf_weight_dict(total_num, number_dict):
        # Functional programming approach with map
        keys, counts = zip(*number_dict.items())
        log_weights = map(lambda c: np.log((total_num + 1) / (c + 1)), counts)
        return dict(zip(keys, log_weights))

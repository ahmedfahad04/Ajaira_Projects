import numpy as np
from gensim import matutils
from numpy import dot, array


class VectorUtil:
    @staticmethod
    def similarity(vector_1, vector_2):
        return dot(matutils.unitvec(vector_1), matutils.unitvec(vector_2))

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        # Matrix formulation using outer operations
        v1_unit = vector_1 / np.linalg.norm(vector_1)
        vectors_unit = vectors_all / np.expand_dims(np.linalg.norm(vectors_all, axis=1), axis=1)
        return vectors_unit @ v1_unit

    @staticmethod
    def n_similarity(vector_list_1, vector_list_2):
        # Early validation with explicit length checks
        len1, len2 = len(vector_list_1), len(vector_list_2)
        if len1 == 0 or len2 == 0:
            raise ZeroDivisionError('At least one of the passed list is empty.')

        # Reduce-style computation
        centroid1 = np.add.reduce(array(vector_list_1)) / len1
        centroid2 = np.add.reduce(array(vector_list_2)) / len2
        
        return dot(matutils.unitvec(centroid1), matutils.unitvec(centroid2))

    @staticmethod
    def compute_idf_weight_dict(total_num, number_dict):
        # NumPy structured approach with fromiter
        items = list(number_dict.items())
        keys, counts = zip(*items) if items else ([], [])
        
        if not counts:
            return {}
            
        count_array = np.fromiter(counts, dtype=float)
        weight_array = np.log((total_num + 1) / (count_array + 1))
        
        return dict(zip(keys, weight_array))

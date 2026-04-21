import numpy as np
from gensim import matutils
from numpy import dot, array


class VectorOperations:
    @staticmethod
    def similarity_calculation(vector_i, vector_j):
        return dot(matutils.unitvec(vector_i), matutils.unitvec(vector_j))

    @staticmethod
    def cosine_similarity_calculation(vector_i, all_vectors):
        norm_i = np.linalg.norm(vector_i)
        norms_all = np.linalg.norm(all_vectors, axis=1)
        dot_products = dot(all_vectors, vector_i)
        return dot_products / (norm_i * norms_all)

    @staticmethod
    def n_similarity_calculation(vector_list_a, vector_list_b):
        if not (len(vector_list_a) and len(vector_list_b)):
            raise ZeroDivisionError('At least one of the passed lists is empty.')

        average_vector_a = array(vector_list_a).mean(axis=0)
        average_vector_b = array(vector_list_b).mean(axis=0)
        return dot(matutils.unitvec(average_vector_a), matutils.unitvec(average_vector_b))

    @staticmethod
    def idf_weight_derivation(total_count, count_dict):
        index_key_map = {}

        index = 0
        count_values = []
        for key, count in count_dict.items():
            index_key_map[index] = key
            count_values.append(count)
            index += 1

        counts_array = np.array(count_values)
        counts_array += 1
        idf_array = np.log((total_count + 1) / counts_array)
        idf_weights = {}

        for idx, weight in enumerate(idf_array):
            key = index_key_map[idx]
            idf_weights[key] = weight

        return idf_weights

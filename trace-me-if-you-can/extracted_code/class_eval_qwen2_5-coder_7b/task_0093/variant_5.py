import numpy as np
from gensim import matutils
from numpy import dot, array


class SemanticUtils:
    @staticmethod
    def similarity(vector_x, vector_y):
        return dot(matutils.unitvec(vector_x), matutils.unitvec(vector_y))

    @staticmethod
    def cosine_similarities(vector_x, all_vectors):
        norm_x = np.linalg.norm(vector_x)
        norms_all = np.linalg.norm(all_vectors, axis=1)
        dot_products = dot(all_vectors, vector_x)
        return dot_products / (norm_x * norms_all)

    @staticmethod
    def n_similarity(vector_list_1, vector_list_2):
        if not (len(vector_list_1) and len(vector_list_2)):
            raise ZeroDivisionError('At least one of the passed lists is empty.')

        avg_vec1 = array(vector_list_1).mean(axis=0)
        avg_vec2 = array(vector_list_2).mean(axis=0)
        return dot(matutils.unitvec(avg_vec1), matutils.unitvec(avg_vec2))

    @staticmethod
    def idf_weights(total_count, count_dict):
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

import numpy as np
from gensim import matutils
from numpy import dot, array


class VectorHandler:
    @staticmethod
    def calc_similarity(vector_a, vector_b):
        return dot(matutils.unitvec(vector_a), matutils.unitvec(vector_b))

    @staticmethod
    def calc_cosine_similarities(vector_a, all_vectors):
        norm_a = np.linalg.norm(vector_a)
        norms_all = np.linalg.norm(all_vectors, axis=1)
        dot_products = dot(all_vectors, vector_a)
        return dot_products / (norm_a * norms_all)

    @staticmethod
    def calculate_n_similarity(list_a, list_b):
        if not (len(list_a) and len(list_b)):
            raise ZeroDivisionError('At least one of the passed lists is empty.')

        avg_vec_a = array(list_a).mean(axis=0)
        avg_vec_b = array(list_b).mean(axis=0)
        return dot(matutils.unitvec(avg_vec_a), matutils.unitvec(avg_vec_b))

    @staticmethod
    def get_idf_weights(total_count, count_dict):
        index_to_key = {}

        index = 0
        count_list = []
        for key, count in count_dict.items():
            index_to_key[index] = key
            count_list.append(count)
            index += 1

        count_array = np.array(count_list)
        count_array += 1
        idf_array = np.log((total_count + 1) / count_array)
        idf_weights = {}

        for idx, weight in enumerate(idf_array):
            key = index_to_key[idx]
            idf_weights[key] = weight

        return idf_weights

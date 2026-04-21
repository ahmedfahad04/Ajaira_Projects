import numpy as np
from gensim import matutils
from numpy import dot, array


class VectorUtility:
    @staticmethod
    def calculate_similarity(vec1, vec2):
        return dot(matutils.unitvec(vec1), matutils.unitvec(vec2))

    @staticmethod
    def calculate_cosine_similarities(vec1, all_vecs):
        norm1 = np.linalg.norm(vec1)
        norms_all = np.linalg.norm(all_vecs, axis=1)
        dot_prods = dot(all_vecs, vec1)
        return dot_prods / (norm1 * norms_all)

    @staticmethod
    def compute_n_similarity(vec_list1, vec_list2):
        if not (len(vec_list1) and len(vec_list2)):
            raise ZeroDivisionError('At least one of the passed lists is empty.')

        avg_vec1 = array(vec_list1).mean(axis=0)
        avg_vec2 = array(vec_list2).mean(axis=0)
        return dot(matutils.unitvec(avg_vec1), matutils.unitvec(avg_vec2))

    @staticmethod
    def derive_idf_weights(total_count, count_dict):
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

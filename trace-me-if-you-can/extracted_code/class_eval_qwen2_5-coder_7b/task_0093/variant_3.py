import numpy as np
from gensim import matutils
from numpy import dot, array


class SimilarityCalculator:
    @staticmethod
    def compute_similarity(vector_x, vector_y):
        return dot(matutils.unitvec(vector_x), matutils.unitvec(vector_y))

    @staticmethod
    def compute_cosine_similarities(vector_x, all_vectors):
        norm_x = np.linalg.norm(vector_x)
        norms_all = np.linalg.norm(all_vectors, axis=1)
        dot_products = dot(all_vectors, vector_x)
        return dot_products / (norm_x * norms_all)

    @staticmethod
    def calculate_n_similarity(vector_list_1, vector_list_2):
        if not (len(vector_list_1) and len(vector_list_2)):
            raise ZeroDivisionError('At least one of the passed lists is empty.')

        mean_vector_1 = array(vector_list_1).mean(axis=0)
        mean_vector_2 = array(vector_list_2).mean(axis=0)
        return dot(matutils.unitvec(mean_vector_1), matutils.unitvec(mean_vector_2))

    @staticmethod
    def derive_idf_weight_dictionary(total_number, count_dictionary):
        index_to_key = {}

        index = 0
        count_values = []
        for key, count in count_dictionary.items():
            index_to_key[index] = key
            count_values.append(count)
            index += 1

        counts_array = np.array(count_values)
        counts_array += 1
        idf_array = np.log((total_number + 1) / counts_array)
        idf_weights = {}

        for idx, weight in enumerate(idf_array):
            key = index_to_key[idx]
            idf_weights[key] = weight

        return idf_weights

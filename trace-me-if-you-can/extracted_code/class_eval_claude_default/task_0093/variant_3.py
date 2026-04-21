import numpy as np
from gensim import matutils
from numpy import dot, array


class VectorUtil:
    @staticmethod
    def similarity(vector_1, vector_2):
        return dot(matutils.unitvec(vector_1), matutils.unitvec(vector_2))

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

        return dot(matutils.unitvec(array(vector_list_1).mean(axis=0)),
                   matutils.unitvec(array(vector_list_2).mean(axis=0)))

    @staticmethod
    def compute_idf_weight_dict(total_num, number_dict):
        # Iterative approach with pre-allocated result
        result = {}
        
        for key, count in number_dict.items():
            idf_weight = np.log((total_num + 1) / (count + 1))
            result[key] = idf_weight
            
        return result

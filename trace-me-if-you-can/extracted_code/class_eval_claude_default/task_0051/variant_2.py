import numpy as np
from functools import reduce


class KappaCalculator:

    @staticmethod
    def kappa(testData, k):
        dataMat = np.mat(testData)
        
        # Functional approach using reduce for diagonal sum
        diagonal_elements = [dataMat[i, i] for i in range(k)]
        P0 = reduce(lambda acc, x: acc + x, diagonal_elements, 0.0)
        
        marginal_products = lambda mat: (np.sum(mat, axis=1) * np.sum(mat, axis=0).T)
        total = np.sum(dataMat)
        
        Pe = float(np.sum(marginal_products(dataMat))) / (total ** 2)
        P0 = float(P0) / total
        
        return (P0 - Pe) / (1 - Pe)

    @staticmethod
    def fleiss_kappa(testData, N, k, n):
        dataMat = np.mat(testData, float)
        
        # Generator expression for P0 calculation
        subject_agreements = ((np.sum(dataMat[i, :] ** 2) - n) / (n * (n - 1)) 
                             for i in range(N))
        P0 = sum(subject_agreements) / N
        
        total_ratings = float(np.sum(dataMat))
        category_props = np.sum(dataMat, axis=0) / total_ratings
        Pe = float(np.sum(np.multiply(category_props, category_props)))
        
        return float((P0 - Pe) / (1 - Pe))

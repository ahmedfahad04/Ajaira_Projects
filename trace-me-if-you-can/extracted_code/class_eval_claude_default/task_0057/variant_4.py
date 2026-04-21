import numpy as np
from functools import singledispatch


class MetricsCalculator2:
    def __init__(self):
        pass

    @staticmethod
    def mrr(data):
        return MetricsCalculator2._process_metric(data, MetricsCalculator2._mrr_kernel)

    @staticmethod
    def map(data):
        return MetricsCalculator2._process_metric(data, MetricsCalculator2._map_kernel)

    @staticmethod
    def _process_metric(data, kernel_func):
        """Generic metric processing pipeline"""
        # Type validation
        if not isinstance(data, (list, tuple)):
            raise Exception("the input must be a tuple([0,...,1,...],int) or a iteration of list of tuple")
        
        # Handle empty data
        if len(data) == 0:
            return 0.0, [0.0]
        
        # Process based on data structure
        if isinstance(data, tuple):
            # Single query case
            result = kernel_func(*data)
            return result, [result]
        else:
            # Multiple queries case
            results = [kernel_func(sub_list, total_num) for sub_list, total_num in data]
            return np.mean(results), results

    @staticmethod
    def _mrr_kernel(sub_list, total_num):
        """Core MRR computation for a single query"""
        if total_num == 0:
            return 0.0
        
        # Convert to numpy and create position-based weights
        relevance_array = np.asarray(sub_list)
        position_weights = 1.0 / (np.arange(len(relevance_array)) + 1)
        
        # Calculate reciprocal rank for relevant items
        weighted_relevance = relevance_array * position_weights
        
        # Return the reciprocal rank of first relevant item
        nonzero_indices = np.nonzero(weighted_relevance)[0]
        return weighted_relevance[nonzero_indices[0]] if len(nonzero_indices) > 0 else 0.0

    @staticmethod
    def _map_kernel(sub_list, total_num):
        """Core MAP computation for a single query"""
        if total_num == 0:
            return 0.0
        
        # Convert to numpy and create position-based weights
        relevance_array = np.asarray(sub_list)
        position_weights = 1.0 / (np.arange(len(relevance_array)) + 1)
        
        # Build precision-at-k array using enumerate and conditional logic
        precision_contributions = []
        relevant_count = 0
        
        for position, is_relevant in enumerate(relevance_array):
            if is_relevant:
                relevant_count += 1
                precision_contributions.append(relevant_count * position_weights[position])
            else:
                precision_contributions.append(0.0)
        
        return np.sum(precision_contributions) / total_num

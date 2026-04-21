import numpy as np


class MetricsCalculator2:
    def __init__(self):
        pass

    @staticmethod
    def mrr(data):
        def _validate_input(data):
            if not isinstance(data, (list, tuple)):
                raise Exception("the input must be a tuple([0,...,1,...],int) or a iteration of list of tuple")
            return len(data) == 0

        def _calculate_single_mrr(sub_list, total_num):
            if total_num == 0:
                return 0.0
            
            sub_array = np.array(sub_list)
            positions = np.arange(1, len(sub_array) + 1)
            reciprocal_ranks = 1.0 / positions
            
            relevant_positions = sub_array * reciprocal_ranks
            first_relevant = relevant_positions[relevant_positions > 0]
            
            return first_relevant[0] if len(first_relevant) > 0 else 0.0

        if _validate_input(data):
            return 0.0, [0.0]

        if isinstance(data, tuple):
            sub_list, total_num = data
            mrr_score = _calculate_single_mrr(sub_list, total_num)
            return mrr_score, [mrr_score]
        
        # Handle list of tuples
        results = [_calculate_single_mrr(sub_list, total_num) for sub_list, total_num in data]
        return np.mean(results), results

    @staticmethod
    def map(data):
        def _validate_input(data):
            if not isinstance(data, (list, tuple)):
                raise Exception("the input must be a tuple([0,...,1,...],int) or a iteration of list of tuple")
            return len(data) == 0

        def _calculate_single_map(sub_list, total_num):
            if total_num == 0:
                return 0.0
            
            sub_array = np.array(sub_list)
            positions = np.arange(1, len(sub_array) + 1)
            reciprocal_ranks = 1.0 / positions
            
            precision_at_k = np.cumsum(sub_array) * sub_array
            return np.sum(precision_at_k * reciprocal_ranks) / total_num

        if _validate_input(data):
            return 0.0, [0.0]

        if isinstance(data, tuple):
            sub_list, total_num = data
            map_score = _calculate_single_map(sub_list, total_num)
            return map_score, [map_score]
        
        # Handle list of tuples
        results = [_calculate_single_map(sub_list, total_num) for sub_list, total_num in data]
        return np.mean(results), results

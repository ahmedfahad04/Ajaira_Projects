import numpy as np


class MetricsCalculator2:
    def __init__(self):
        pass

    @staticmethod
    def mrr(data):
        # Early validation and normalization
        if not isinstance(data, (list, tuple)):
            raise Exception("the input must be a tuple([0,...,1,...],int) or a iteration of list of tuple")

        if len(data) == 0:
            return 0.0, [0.0]

        # Normalize input to always be a list of tuples
        normalized_data = [data] if isinstance(data, tuple) else data
        
        mrr_scores = []
        for item in normalized_data:
            sub_list, total_num = item
            
            if total_num == 0:
                mrr_scores.append(0.0)
                continue
            
            # Vectorized computation using numpy operations
            relevance_vector = np.array(sub_list)
            rank_vector = np.arange(1, len(relevance_vector) + 1)
            reciprocal_ranks = np.divide(1.0, rank_vector, where=rank_vector!=0)
            
            # Find first relevant item using argmax on boolean array
            relevant_mask = relevance_vector > 0
            if np.any(relevant_mask):
                first_relevant_idx = np.argmax(relevant_mask)
                mrr_scores.append(reciprocal_ranks[first_relevant_idx])
            else:
                mrr_scores.append(0.0)
        
        # Return format depends on original input type
        if isinstance(data, tuple):
            return mrr_scores[0], mrr_scores
        else:
            return np.mean(mrr_scores), mrr_scores

    @staticmethod
    def map(data):
        # Early validation and normalization
        if not isinstance(data, (list, tuple)):
            raise Exception("the input must be a tuple([0,...,1,...],int) or a iteration of list of tuple")

        if len(data) == 0:
            return 0.0, [0.0]

        # Normalize input to always be a list of tuples
        normalized_data = [data] if isinstance(data, tuple) else data
        
        map_scores = []
        for item in normalized_data:
            sub_list, total_num = item
            
            if total_num == 0:
                map_scores.append(0.0)
                continue
            
            # Vectorized computation using numpy operations
            relevance_vector = np.array(sub_list)
            rank_vector = np.arange(1, len(relevance_vector) + 1)
            reciprocal_ranks = np.divide(1.0, rank_vector)
            
            # Calculate precision at each position using cumulative sum
            cumulative_relevant = np.cumsum(relevance_vector)
            precision_at_k = cumulative_relevant / rank_vector
            
            # Average precision calculation
            ap_numerator = np.sum(precision_at_k * relevance_vector * reciprocal_ranks)
            map_scores.append(ap_numerator / total_num)
        
        # Return format depends on original input type
        if isinstance(data, tuple):
            return map_scores[0], map_scores
        else:
            return np.mean(map_scores), map_scores

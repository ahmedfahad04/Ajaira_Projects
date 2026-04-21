import numpy as np


class MetricsCalculator2:
    def __init__(self):
        pass

    @classmethod
    def _validate_and_extract(cls, data):
        """Unified input validation and extraction logic"""
        if not isinstance(data, (list, tuple)):
            raise Exception("the input must be a tuple([0,...,1,...],int) or a iteration of list of tuple")
        
        if len(data) == 0:
            return [], True  # empty_flag = True
        
        if isinstance(data, tuple):
            return [data], False  # single item wrapped in list
        
        return data, False  # already a list

    @staticmethod
    def mrr(data):
        items, is_empty = MetricsCalculator2._validate_and_extract(data)
        
        if is_empty:
            return 0.0, [0.0]
        
        # Process all items using list comprehension with conditional logic
        scores = [
            0.0 if total_num == 0 
            else next(
                (1.0 / (idx + 1) for idx, val in enumerate(sub_list) if val > 0), 
                0.0
            )
            for sub_list, total_num in items
        ]
        
        # Return appropriate format based on original input
        if isinstance(data, tuple):
            return scores[0], scores
        return np.mean(scores), scores

    @staticmethod  
    def map(data):
        items, is_empty = MetricsCalculator2._validate_and_extract(data)
        
        if is_empty:
            return 0.0, [0.0]
        
        def compute_average_precision(sub_list, total_num):
            if total_num == 0:
                return 0.0
            
            # Use generator expression with enumerate for efficiency
            precision_sum = sum(
                (sum(sub_list[:i+1]) / (i+1)) / (i+1) 
                for i, relevance in enumerate(sub_list) 
                if relevance > 0
            )
            
            return precision_sum / total_num
        
        # Process all items using the helper function
        scores = [compute_average_precision(sub_list, total_num) for sub_list, total_num in items]
        
        # Return appropriate format based on original input
        if isinstance(data, tuple):
            return scores[0], scores
        return np.mean(scores), scores

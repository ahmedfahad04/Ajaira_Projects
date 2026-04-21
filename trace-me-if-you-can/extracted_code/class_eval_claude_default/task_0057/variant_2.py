import numpy as np
from typing import Union, List, Tuple


class MetricsCalculator2:
    def __init__(self):
        pass

    @staticmethod
    def mrr(data: Union[List, Tuple]) -> Tuple[float, List[float]]:
        # Input validation using dictionary dispatch
        handlers = {
            list: MetricsCalculator2._handle_list_mrr,
            tuple: MetricsCalculator2._handle_tuple_mrr
        }
        
        data_type = type(data)
        if data_type not in handlers:
            raise Exception("the input must be a tuple([0,...,1,...],int) or a iteration of list of tuple")
        
        if len(data) == 0:
            return 0.0, [0.0]
            
        return handlers[data_type](data)

    @staticmethod
    def _handle_tuple_mrr(data: Tuple) -> Tuple[float, List[float]]:
        sub_list, total_num = data
        if total_num == 0:
            return 0.0, [0.0]
        
        mrr_value = MetricsCalculator2._compute_mrr_score(sub_list)
        return mrr_value, [mrr_value]

    @staticmethod
    def _handle_list_mrr(data: List) -> Tuple[float, List[float]]:
        scores = []
        for sub_list, total_num in data:
            score = 0.0 if total_num == 0 else MetricsCalculator2._compute_mrr_score(sub_list)
            scores.append(score)
        return np.mean(scores), scores

    @staticmethod
    def _compute_mrr_score(sub_list) -> float:
        sub_array = np.array(sub_list)
        rank_weights = 1.0 / np.arange(1, len(sub_array) + 1)
        weighted_relevance = sub_array * rank_weights
        
        relevant_indices = np.where(weighted_relevance > 0)[0]
        return weighted_relevance[relevant_indices[0]] if len(relevant_indices) > 0 else 0.0

    @staticmethod
    def map(data: Union[List, Tuple]) -> Tuple[float, List[float]]:
        # Input validation using dictionary dispatch
        handlers = {
            list: MetricsCalculator2._handle_list_map,
            tuple: MetricsCalculator2._handle_tuple_map
        }
        
        data_type = type(data)
        if data_type not in handlers:
            raise Exception("the input must be a tuple([0,...,1,...],int) or a iteration of list of tuple")
        
        if len(data) == 0:
            return 0.0, [0.0]
            
        return handlers[data_type](data)

    @staticmethod
    def _handle_tuple_map(data: Tuple) -> Tuple[float, List[float]]:
        sub_list, total_num = data
        if total_num == 0:
            return 0.0, [0.0]
        
        map_value = MetricsCalculator2._compute_map_score(sub_list, total_num)
        return map_value, [map_value]

    @staticmethod
    def _handle_list_map(data: List) -> Tuple[float, List[float]]:
        scores = []
        for sub_list, total_num in data:
            score = 0.0 if total_num == 0 else MetricsCalculator2._compute_map_score(sub_list, total_num)
            scores.append(score)
        return np.mean(scores), scores

    @staticmethod
    def _compute_map_score(sub_list, total_num: int) -> float:
        sub_array = np.array(sub_list)
        rank_weights = 1.0 / np.arange(1, len(sub_array) + 1)
        
        cumulative_precision = np.cumsum(sub_array)
        precision_at_relevant = cumulative_precision * sub_array
        
        return np.sum(precision_at_relevant * rank_weights) / total_num

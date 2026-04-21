import numpy as np

class MetricsCalculator:
    @staticmethod
    def calculate_mrr(data):
        if not isinstance(data, (list, tuple)):
            raise ValueError("Input must be a list or tuple of tuples")
        if not data:
            return 0.0, [0.0]
        
        sub_list, total_num = data if isinstance(data, tuple) else data[0]
        sub_list = np.array(sub_list)
        if total_num == 0:
            return 0.0, [0.0]
        
        ranking_array = 1.0 / np.arange(len(sub_list)) + 1
        mr = 0.0
        for index, value in enumerate(sub_list * ranking_array):
            if value > 0:
                mr = value
                break
        return mr, [mr]

    @staticmethod
    def calculate_map(data):
        if not isinstance(data, (list, tuple)):
            raise ValueError("Input must be a list or tuple of tuples")
        if not data:
            return 0.0, [0.0]
        
        if isinstance(data, tuple):
            sub_list, total_num = data
        else:
            sub_list, total_num = data[0]
        sub_list = np.array(sub_list)
        if total_num == 0:
            return 0.0, [0.0]
        
        ranking_array = 1.0 / np.arange(len(sub_list)) + 1
        right_ranking_list = []
        count = 1
        for value in sub_list:
            if value == 0:
                right_ranking_list.append(0)
            else:
                right_ranking_list.append(count)
                count += 1
        
        ap = np.sum(np.array(right_ranking_list) * ranking_array) / total_num
        return ap, [ap]

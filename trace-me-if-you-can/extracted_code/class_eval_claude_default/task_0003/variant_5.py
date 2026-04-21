import itertools


class ArrangementCalculator:
    def __init__(self, datas):
        self.datas = datas

    @staticmethod
    def count(n, m=None):
        m = m if m is not None else n
        
        def permutation_formula(total, select):
            numerator = 1
            for x in range(total, total - select, -1):
                numerator *= x
            return numerator
        
        return permutation_formula(n, m)

    @staticmethod
    def count_all(n):
        def sum_all_arrangements(total):
            running_sum = 0
            arrangement_size = 1
            while arrangement_size <= total:
                running_sum += ArrangementCalculator.count(total, arrangement_size)
                arrangement_size += 1
            return running_sum
        
        return sum_all_arrangements(n)

    def select(self, m=None):
        def generate_arrangements(items, size):
            size = len(items) if size is None else size
            arrangement_list = []
            for arrangement in itertools.permutations(items, size):
                arrangement_list.append(list(arrangement))
            return arrangement_list
        
        return generate_arrangements(self.datas, m)

    def select_all(self):
        def collect_all_sizes(items):
            all_arrangements = []
            max_size = len(items)
            current_size = 1
            while current_size <= max_size:
                all_arrangements.extend(self.select(current_size))
                current_size += 1
            return all_arrangements
        
        return collect_all_sizes(self.datas)

    @staticmethod
    def factorial(n):
        def calculate_factorial(num):
            if num <= 1:
                return 1
            factorial_result = 1
            counter = 2
            while counter <= num:
                factorial_result *= counter
                counter += 1
            return factorial_result
        
        return calculate_factorial(n)

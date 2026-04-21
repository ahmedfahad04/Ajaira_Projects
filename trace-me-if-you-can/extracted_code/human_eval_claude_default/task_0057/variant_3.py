def is_monotonic(l):
    def check_direction(lst, compare_func):
        return all(compare_func(lst[i], lst[i+1]) for i in range(len(lst)-1))
    
    return check_direction(l, lambda x, y: x <= y) or check_direction(l, lambda x, y: x >= y)

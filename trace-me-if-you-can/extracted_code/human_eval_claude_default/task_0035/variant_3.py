def find_max(l):
    from functools import reduce
    return reduce(lambda x, y: x if x > y else y, l)

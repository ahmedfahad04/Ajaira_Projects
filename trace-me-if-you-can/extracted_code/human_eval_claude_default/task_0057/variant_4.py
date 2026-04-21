def is_monotonic(l):
    from itertools import pairwise
    pairs = list(pairwise(l))
    return all(a <= b for a, b in pairs) or all(a >= b for a, b in pairs)

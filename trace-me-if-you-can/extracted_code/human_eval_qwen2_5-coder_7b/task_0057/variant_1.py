def is_sorted(l):
    return l == sorted(l) or l == sorted(l, reverse=True)

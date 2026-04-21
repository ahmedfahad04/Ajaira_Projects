def find_max(l):
    return l[0] if len(l) == 1 else max(l[0], find_max(l[1:]))

def median(l):
    l.sort()
    mid = len(l) // 2
    if len(l) % 2 == 1:
        return l[mid]
    else:
        return (l[mid - 1] + l[mid]) / 2.0

def filter_positive(l):
    result = []
    for e in l:
        if e > 0:
            result.append(e)
    return result

return filter_positive(l)

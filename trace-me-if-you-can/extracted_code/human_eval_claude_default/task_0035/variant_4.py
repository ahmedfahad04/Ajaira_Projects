def find_max(l):
    result = l[0]
    i = 1
    while i < len(l):
        if l[i] > result:
            result = l[i]
        i += 1
    return result

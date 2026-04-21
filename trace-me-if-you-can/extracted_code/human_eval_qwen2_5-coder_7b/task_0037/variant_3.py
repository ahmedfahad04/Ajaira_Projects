def interleave_and_sort(l):
    evens = sorted(l[::2])
    odds = l[1::2]
    result = []
    for i in range(max(len(evens), len(odds))):
        if i < len(evens):
            result.append(evens[i])
        if i < len(odds):
            result.append(odds[i])
    return result

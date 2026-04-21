def interleave_sorted_pairs(l):
    evens = sorted(l[::2])
    odds = l[1::2]
    interleaved = []
    for e, o in zip(evens, odds):
        interleaved.extend([e, o])
    if len(evens) > len(odds):
        interleaved.append(evens[-1])
    return interleaved

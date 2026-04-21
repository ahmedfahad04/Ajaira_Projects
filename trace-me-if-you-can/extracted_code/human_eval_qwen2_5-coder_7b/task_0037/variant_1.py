def interleave_sorted_lists(l):
    evens = sorted(l[::2])
    odds = l[1::2]
    interleaved = [None] * (len(evens) + len(odds))
    interleaved[::2] = evens
    interleaved[1::2] = odds
    return interleaved

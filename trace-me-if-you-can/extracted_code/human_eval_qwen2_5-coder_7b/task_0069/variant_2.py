def identify_min_freq(lst):
    freqs = [0] * (max(lst) + 1)
    for item in lst:
        freqs[item] += 1

    result = -1
    for index, value in enumerate(freqs):
        if value >= index:
            result = index

    return result

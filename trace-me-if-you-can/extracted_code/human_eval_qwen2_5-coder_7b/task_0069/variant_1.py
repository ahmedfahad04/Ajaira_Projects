def find_min_freq(lst):
    freq_map = {}
    for num in lst:
        if num in freq_map:
            freq_map[num] += 1
        else:
            freq_map[num] = 1

    min_val = -1
    for num, count in freq_map.items():
        if count >= num:
            min_val = num

    return min_val

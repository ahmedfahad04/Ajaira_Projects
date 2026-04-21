def compute_min_frequency(lst):
    count_dict = {}
    for elem in lst:
        if elem in count_dict:
            count_dict[elem] += 1
        else:
            count_dict[elem] = 1

    output = -1
    for k, v in count_dict.items():
        if v >= k:
            output = k

    return output

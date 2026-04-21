def find_zero_sum_pairs(lst):
    for idx, elem1 in enumerate(lst):
        for idx2 in range(idx + 1, len(lst)):
            if elem1 + lst[idx2] == 0:
                return True
    return False

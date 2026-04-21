def get_median(lst):
    lst = sorted(lst)
    median_index = len(lst) // 2
    if len(lst) % 2:
        return lst[median_index]
    else:
        return (lst[median_index - 1] + lst[median_index]) / 2.0

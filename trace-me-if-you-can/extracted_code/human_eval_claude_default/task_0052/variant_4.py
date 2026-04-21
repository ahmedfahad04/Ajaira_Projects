def check_all_less_than(lst, threshold):
    if not lst:
        return True
    return lst[0] < threshold and check_all_less_than(lst[1:], threshold)

return check_all_less_than(l, t)

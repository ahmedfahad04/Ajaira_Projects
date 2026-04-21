def get_second_smallest(lst):
    unique_lst = list(dict.fromkeys(lst))
    unique_lst.sort()
    return None if len(unique_lst) < 2 else unique_lst[1]

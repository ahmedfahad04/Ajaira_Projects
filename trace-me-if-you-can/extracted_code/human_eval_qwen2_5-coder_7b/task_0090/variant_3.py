def get_penultimate_smallest(lst):
    unique_set = set(lst)
    if len(unique_set) < 2:
        return None
    sorted_list = sorted(unique_set)
    return sorted_list[1]

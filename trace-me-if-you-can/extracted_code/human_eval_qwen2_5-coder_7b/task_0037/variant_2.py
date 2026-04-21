def merge_and_sort_lists(lst):
    even_indices = range(0, len(lst), 2)
    odd_indices = range(1, len(lst), 2)
    sorted_evens = sorted(lst[i] for i in even_indices)
    merged_list = []
    for e, o in zip(sorted_evens, lst[odd_indices]):
        merged_list.extend([e, o])
    if len(sorted_evens) > len(lst[odd_indices]):
        merged_list.append(sorted_evens[-1])
    return merged_list

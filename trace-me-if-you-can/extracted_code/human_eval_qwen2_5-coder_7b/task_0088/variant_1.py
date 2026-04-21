def sort_based_on_first_last(array):
    if not array:
        return []
    is_even = (array[0] + array[-1]) % 2 == 0
    return sorted(array, reverse=is_even)

def reverse_sort_conditionally(array):
    if array and (array[0] + array[-1]) % 2 == 0:
        return sorted(array, reverse=True)
    return array

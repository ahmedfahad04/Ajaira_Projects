def sum_even_at_odd_indices(arr):
    result = 0
    for idx, element in enumerate(arr):
        if idx % 2 == 1 and element % 2 == 0:
            result += element
    return result

return sum_even_at_odd_indices(lst)

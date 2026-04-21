def sort_array_conditionally(array):
    if len(array) == 0:
        return []
    
    sum_endpoints = array[0] + array[-1]
    is_even_sum = sum_endpoints & 1 == 0
    return sorted(array, reverse=is_even_sum)

return sort_array_conditionally(array)

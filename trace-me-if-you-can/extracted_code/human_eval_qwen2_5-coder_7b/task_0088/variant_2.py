if array:
    is_descending = (array[0] + array[-1]) % 2 == 0
    return sorted(array, reverse=is_descending)
return []

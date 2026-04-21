if not array:
    return []
first_last_sum = array[0] + array[-1]
descending = first_last_sum % 2 == 0
return sorted(array, reverse=descending)

if len(array) > 0:
    should_reverse = (array[0] + array[-1]) % 2 == 0
    return sorted(array, reverse=should_reverse)
return []

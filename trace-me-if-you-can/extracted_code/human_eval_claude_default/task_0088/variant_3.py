try:
    sort_descending = not bool((array[0] + array[-1]) % 2)
    return sorted(array, reverse=sort_descending)
except IndexError:
    return []

def has_zero_sum_pair(arr, start=0):
    if start >= len(arr) - 1:
        return False
    
    current = arr[start]
    for i in range(start + 1, len(arr)):
        if current + arr[i] == 0:
            return True
    
    return has_zero_sum_pair(arr, start + 1)

return has_zero_sum_pair(l)

def running_maximum(numbers):
    result = []
    
    for i, n in enumerate(numbers):
        if i == 0:
            current_max = n
        else:
            current_max = max(result[i-1], n)
        result.append(current_max)
    
    return result

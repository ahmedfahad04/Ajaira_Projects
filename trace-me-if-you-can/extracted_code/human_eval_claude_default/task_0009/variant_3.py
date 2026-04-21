def running_maximum(numbers):
    if not numbers:
        return []
    
    result = [numbers[0]]
    running_max = numbers[0]
    
    for i in range(1, len(numbers)):
        running_max = max(running_max, numbers[i])
        result.append(running_max)
    
    return result

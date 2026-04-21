def normalize(numbers):
    if not numbers:
        return []
    
    min_val = numbers[0]
    max_val = numbers[0]
    
    for num in numbers:
        if num < min_val:
            min_val = num
        if num > max_val:
            max_val = num
    
    range_val = max_val - min_val
    if range_val == 0:
        return [0.0] * len(numbers)
    
    result = []
    for num in numbers:
        result.append((num - min_val) / range_val)
    return result

# Variant 4: Index-based iteration approach
def normalize(numbers):
    if not numbers:
        return []
    
    n = len(numbers)
    min_val = max_val = numbers[0]
    
    # Single pass to find extremes
    for i in range(1, n):
        if numbers[i] < min_val:
            min_val = numbers[i]
        elif numbers[i] > max_val:
            max_val = numbers[i]
    
    range_diff = max_val - min_val
    if range_diff == 0:
        return [0.0] * n
    
    normalized = [0.0] * n
    for i in range(n):
        normalized[i] = (numbers[i] - min_val) / range_diff
    
    return normalized

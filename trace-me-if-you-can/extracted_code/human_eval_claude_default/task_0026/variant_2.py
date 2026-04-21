def filter_unique_occurrences(numbers):
    frequency_map = {}
    for num in numbers:
        frequency_map[num] = frequency_map.get(num, 0) + 1
    
    result = []
    for num in numbers:
        if frequency_map[num] <= 1:
            result.append(num)
    
    return result

def filter_unique_occurrences(numbers):
    if not numbers:
        return []
    
    def count_occurrences(arr, target):
        return sum(1 for x in arr if x == target)
    
    unique_elements = []
    processed = set()
    
    for num in numbers:
        if num not in processed:
            if count_occurrences(numbers, num) <= 1:
                unique_elements.extend([num] * numbers.count(num))
            processed.add(num)
    
    # Maintain original order
    result = []
    for num in numbers:
        if count_occurrences(numbers, num) <= 1:
            result.append(num)
    
    return result

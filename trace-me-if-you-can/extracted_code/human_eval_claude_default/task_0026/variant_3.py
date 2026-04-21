def filter_unique_occurrences(numbers):
    from functools import reduce
    
    def count_and_filter(acc, num):
        counts, filtered = acc
        new_count = counts.get(num, 0) + 1
        counts[num] = new_count
        if new_count <= 1:
            filtered.append(num)
        return counts, filtered
    
    _, result = reduce(count_and_filter, numbers, ({}, []))
    
    # Remove elements that appeared more than once
    final_counts = {}
    for num in numbers:
        final_counts[num] = final_counts.get(num, 0) + 1
    
    return [num for num in result if final_counts[num] <= 1]

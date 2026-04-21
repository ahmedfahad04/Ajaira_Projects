from itertools import combinations

def find_closest_pair(numbers):
    if len(numbers) < 2:
        return None
    
    min_distance = float('inf')
    result_pair = None
    
    for pair in combinations(numbers, 2):
        current_distance = abs(pair[0] - pair[1])
        if current_distance < min_distance:
            min_distance = current_distance
            result_pair = tuple(sorted(pair))
    
    return result_pair

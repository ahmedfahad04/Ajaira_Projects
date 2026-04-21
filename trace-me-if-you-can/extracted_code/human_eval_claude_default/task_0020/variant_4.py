# Variant 4: Using heap to maintain closest pairs
import heapq

def find_closest_pair(numbers):
    if len(numbers) < 2:
        return None
    
    distances = []
    
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            distance = abs(numbers[i] - numbers[j])
            pair = tuple(sorted([numbers[i], numbers[j]]))
            heapq.heappush(distances, (distance, pair))
    
    return heapq.heappop(distances)[1]

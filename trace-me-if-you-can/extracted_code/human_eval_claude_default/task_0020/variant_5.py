def find_closest_pair(numbers):
    if len(numbers) < 2:
        return None
    
    def generate_pairs():
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                distance = abs(numbers[i] - numbers[j])
                pair = tuple(sorted([numbers[i], numbers[j]]))
                yield distance, pair
    
    closest_distance = float('inf')
    closest_pair = None
    
    for distance, pair in generate_pairs():
        if distance < closest_distance:
            closest_distance = distance
            closest_pair = pair
            if distance == 0:  # Early termination for identical numbers
                break
    
    return closest_pair

def find_closest_pair(numbers):
    if len(numbers) < 2:
        return None
    
    all_pairs = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            pair = tuple(sorted([numbers[i], numbers[j]]))
            distance = abs(numbers[i] - numbers[j])
            all_pairs.append((pair, distance))
    
    return min(all_pairs, key=lambda x: x[1])[0]

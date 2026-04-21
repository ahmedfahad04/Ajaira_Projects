def find_closest_pair(numbers):
    min_distance = float('inf')
    closest_pair = None

    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            current_distance = abs(numbers[i] - numbers[j])
            if current_distance < min_distance:
                min_distance = current_distance
                closest_pair = tuple(sorted([numbers[i], numbers[j]]))

    return closest_pair

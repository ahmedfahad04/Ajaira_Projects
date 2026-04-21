def find_closest_pair(numbers):
    distances = {}
    closest_pair = None
    min_distance = float('inf')

    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            distance = abs(numbers[i] - numbers[j])
            distances[(numbers[i], numbers[j])] = distance
            if distance < min_distance:
                min_distance = distance
                closest_pair = tuple(sorted([numbers[i], numbers[j]]))

    return closest_pair

def find_closest_pair(numbers):
    min_distance = float('inf')
    closest_pair = None

    for i, j in ((x, y) for x in range(len(numbers)) for y in range(x + 1, len(numbers))):
        distance = abs(numbers[i] - numbers[j])
        if distance < min_distance:
            min_distance = distance
            closest_pair = tuple(sorted([numbers[i], numbers[j]]))

    return closest_pair

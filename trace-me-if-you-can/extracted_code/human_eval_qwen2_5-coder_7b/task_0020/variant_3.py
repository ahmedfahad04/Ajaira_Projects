def find_closest_pair(numbers):
    pairs = [(i, j) for i in range(len(numbers)) for j in range(i + 1, len(numbers))]
    distances = sorted(pairs, key=lambda pair: abs(numbers[pair[0]] - numbers[pair[1]]))
    closest_pair = distances[0]
    return tuple(sorted([numbers[closest_pair[0]], numbers[closest_pair[1]]]))

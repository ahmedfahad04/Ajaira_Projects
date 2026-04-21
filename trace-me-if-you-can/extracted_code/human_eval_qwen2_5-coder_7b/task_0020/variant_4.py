def find_closest_pair(numbers):
    distances = [
        ((i, j), abs(numbers[i] - numbers[j]))
        for i in range(len(numbers))
        for j in range(i + 1, len(numbers))
    ]
    closest_pair = min(distances, key=lambda x: x[1])[0]
    return tuple(sorted([numbers[closest_pair[0]], numbers[closest_pair[1]]]))

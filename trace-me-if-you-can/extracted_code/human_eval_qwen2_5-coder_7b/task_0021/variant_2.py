normalized_values = []
    for num in numbers:
        normalized_values.append((num - min(numbers)) / (max(numbers) - min(numbers)))
    return normalized_values

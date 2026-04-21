scaled_values = list(map(lambda x: (x - min(numbers)) / (max(numbers) - min(numbers)), numbers))
    return scaled_values

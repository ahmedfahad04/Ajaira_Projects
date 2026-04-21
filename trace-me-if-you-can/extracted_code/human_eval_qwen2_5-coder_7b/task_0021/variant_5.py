normalized_list = [((n - min(numbers)) / (max(numbers) - min(numbers))) for n in numbers]
    return normalized_list

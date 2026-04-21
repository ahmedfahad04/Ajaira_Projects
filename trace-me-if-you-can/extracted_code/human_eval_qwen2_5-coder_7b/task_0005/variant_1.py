def process_numbers(numbers, delimiter):
    if not numbers:
        return []

    processed = [numbers[0]]
    for n in numbers[1:]:
        processed.extend([n, delimiter])

    return processed

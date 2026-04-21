def intersperse(numbers, delimiter):
    if not numbers:
        return []

    output = []
    for i in range(len(numbers) - 1):
        output.extend([numbers[i], delimiter])
    output.append(numbers[-1])

    return output

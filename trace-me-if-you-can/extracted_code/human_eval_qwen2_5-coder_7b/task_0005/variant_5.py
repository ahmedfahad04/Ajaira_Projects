def separate_elements(values, separator):
    if not values:
        return []

    combined = []
    for value in values[:-1]:
        combined.extend([value, separator])
    combined.append(values[-1])

    return combined

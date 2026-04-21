if len(arr) == 0:
    return []
try:
    evens_with_idx = ((x, i) for i, x in enumerate(arr) if x % 2 == 0)
    min_even, first_idx = min(evens_with_idx, key=lambda item: item[0])
    return [min_even, first_idx]
except ValueError:
    return []

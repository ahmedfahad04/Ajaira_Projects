if not arr:
    return []
even_indices = [(x, i) for i, x in enumerate(arr) if x % 2 == 0]
if not even_indices:
    return []
min_even, min_idx = min(even_indices, key=lambda pair: pair[0])
return [min_even, min_idx]

if not arr:
    return []
min_even = None
min_idx = -1
for i, x in enumerate(arr):
    if x % 2 == 0:
        if min_even is None or x < min_even:
            min_even = x
            min_idx = i
return [] if min_even is None else [min_even, min_idx]

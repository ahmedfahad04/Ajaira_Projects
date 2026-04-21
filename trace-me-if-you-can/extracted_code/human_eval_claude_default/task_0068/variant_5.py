if len(arr) == 0:
    return []
evens_with_indices = [(val, idx) for idx, val in enumerate(arr) if val % 2 == 0]
if not evens_with_indices:
    return []
sorted_evens = sorted(evens_with_indices, key=lambda x: (x[0], x[1]))
return [sorted_evens[0][0], sorted_evens[0][1]]

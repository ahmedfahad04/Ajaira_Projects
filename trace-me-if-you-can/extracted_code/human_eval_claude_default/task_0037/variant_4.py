# Version 4: Using dictionary mapping and reconstruction
even_positions = {i: val for i, val in enumerate(l[::2])}
sorted_even_values = sorted(even_positions.values())

result = [None] * len(l)
# Fill odd positions first
for i in range(1, len(l), 2):
    result[i] = l[i]
# Fill even positions with sorted values
for i, val in enumerate(sorted_even_values):
    result[i * 2] = val
return result

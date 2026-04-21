# Version 1: Using enumerate with nested loops
coords = []
for i, row in enumerate(lst):
    for j, val in enumerate(row):
        if val == x:
            coords.append((i, j))
return sorted(coords, key=lambda coord: (coord[0], -coord[1]))

# Version 4: Using dictionary grouping then flattening
from collections import defaultdict
row_groups = defaultdict(list)
for i in range(len(lst)):
    for j in range(len(lst[i])):
        if lst[i][j] == x:
            row_groups[i].append(j)

coords = []
for row in sorted(row_groups.keys()):
    for col in sorted(row_groups[row], reverse=True):
        coords.append((row, col))
return coords

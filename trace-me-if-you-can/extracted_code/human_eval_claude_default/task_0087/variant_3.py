# Version 3: Using numpy-style approach with manual indexing
result = []
for row_idx in range(len(lst)):
    row = lst[row_idx]
    for col_idx in range(len(row) - 1, -1, -1):  # Reverse order for columns
        if row[col_idx] == x:
            result.append((row_idx, col_idx))
# Sort by row, keeping reverse column order from iteration
return sorted(result, key=lambda item: item[0])

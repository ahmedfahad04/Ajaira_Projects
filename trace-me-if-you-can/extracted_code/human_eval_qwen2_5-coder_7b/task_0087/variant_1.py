def locate_and_sort(lst, x):
    positions = []
    for row_idx, row in enumerate(lst):
        for col_idx, val in enumerate(row):
            if val == x:
                positions.append((row_idx, col_idx))
    sorted_positions = sorted(positions, key=lambda pos: (pos[0], -pos[1]))
    return sorted_positions

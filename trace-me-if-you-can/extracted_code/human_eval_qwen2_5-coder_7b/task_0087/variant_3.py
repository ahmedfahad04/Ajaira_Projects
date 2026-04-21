def get_positions_and_order(lst, x):
    coords = []
    for row in range(len(lst)):
        for col in range(len(lst[row])):
            if lst[row][col] == x:
                coords.append((row, col))
    coords = sorted(coords, key=lambda coord: coord[1], reverse=True)
    coords = sorted(coords, key=lambda coord: coord[0])
    return coords

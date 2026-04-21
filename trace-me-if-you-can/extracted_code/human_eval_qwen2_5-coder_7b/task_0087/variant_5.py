def extract_and_sort(lst, target):
    coordinates = []
    for row_idx, row in enumerate(lst):
        for col_idx, val in enumerate(row):
            if val == target:
                coordinates.append((row_idx, col_idx))
    coordinates.sort(key=lambda coord: coord[0])
    coordinates.sort(key=lambda coord: -coord[1])
    return coordinates

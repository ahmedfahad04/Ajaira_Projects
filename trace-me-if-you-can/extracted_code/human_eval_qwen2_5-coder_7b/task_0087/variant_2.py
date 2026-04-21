def find_and_order(lst, target):
    indices = []
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            if lst[i][j] == target:
                indices.append((i, j))
    indices.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return indices

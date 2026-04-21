def locate_and_arrange(lst, x):
    results = []
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            if lst[i][j] == x:
                results.append((i, j))
    results.sort(key=lambda result: result[1], reverse=True)
    results.sort(key=lambda result: result[0])
    return results

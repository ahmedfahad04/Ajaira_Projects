def count_dissimilarities(arr):
    dissimilarities = 0
    for i in range(len(arr) // 2):
        if arr[i] != arr[len(arr) - i - 1]:
            dissimilarities += 1
    return dissimilarities

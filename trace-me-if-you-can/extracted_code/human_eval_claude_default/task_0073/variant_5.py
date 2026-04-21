def count_mismatches(arr, start=0):
    if start >= len(arr) // 2:
        return 0
    mismatch = 1 if arr[start] != arr[len(arr) - start - 1] else 0
    return mismatch + count_mismatches(arr, start + 1)

return count_mismatches(arr)

def calculate_mismatches(arr):
    mismatch_count = 0
    for i in range(len(arr) // 2):
        if arr[i] != arr[-(i + 1)]:
            mismatch_count += 1
    return mismatch_count

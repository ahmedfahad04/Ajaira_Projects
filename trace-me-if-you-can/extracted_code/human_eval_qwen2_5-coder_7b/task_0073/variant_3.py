def find_differences(arr):
    diff_count = 0
    half_len = len(arr) // 2
    for idx in range(half_len):
        if arr[idx] != arr[half_len - idx - 1]:
            diff_count += 1
    return diff_count

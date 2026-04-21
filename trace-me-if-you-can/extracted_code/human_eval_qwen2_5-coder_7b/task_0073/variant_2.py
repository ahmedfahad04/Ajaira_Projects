mismatches = 0
for idx in range(len(arr) // 2):
    if arr[idx] != arr[len(arr) - idx - 1]:
        mismatches += 1
return mismatches

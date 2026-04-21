left, right = 0, len(arr) - 1
mismatches = 0
while left < right:
    if arr[left] != arr[right]:
        mismatches += 1
    left += 1
    right -= 1
return mismatches

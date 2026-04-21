# Variant 1: Using enumerate with early termination logic
ans = 0
for i, val in enumerate(arr[:len(arr)//2]):
    if val != arr[-(i+1)]:
        ans += 1
return ans

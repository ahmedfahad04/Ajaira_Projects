sorted_l = sorted(enumerate(l), key=lambda x: x[1])
n = len(sorted_l)

for i in range(n - 2):
    left, right = i + 1, n - 1
    while left < right:
        current_sum = sorted_l[i][1] + sorted_l[left][1] + sorted_l[right][1]
        if current_sum == 0:
            # Ensure all three indices are different from original array
            indices = {sorted_l[i][0], sorted_l[left][0], sorted_l[right][0]}
            if len(indices) == 3:
                return True
            left += 1
        elif current_sum < 0:
            left += 1
        else:
            right -= 1
return False

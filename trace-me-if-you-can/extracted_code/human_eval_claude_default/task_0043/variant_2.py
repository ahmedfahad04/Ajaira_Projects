sorted_l = sorted(l)
left, right = 0, len(sorted_l) - 1
while left < right:
    current_sum = sorted_l[left] + sorted_l[right]
    if current_sum == 0:
        return True
    elif current_sum < 0:
        left += 1
    else:
        right -= 1
return False

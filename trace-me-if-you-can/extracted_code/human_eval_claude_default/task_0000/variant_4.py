sorted_nums = sorted(enumerate(numbers), key=lambda x: x[1])
for i in range(len(sorted_nums) - 1):
    for j in range(i + 1, len(sorted_nums)):
        distance = abs(sorted_nums[i][1] - sorted_nums[j][1])
        if distance >= threshold:
            break
        if distance < threshold:
            return True
return False

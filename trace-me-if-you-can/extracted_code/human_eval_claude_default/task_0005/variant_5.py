def helper(nums, delim):
    if len(nums) <= 1:
        return nums[:]
    return [nums[0], delim] + helper(nums[1:], delim)

if not numbers:
    return []
return helper(numbers, delimeter)

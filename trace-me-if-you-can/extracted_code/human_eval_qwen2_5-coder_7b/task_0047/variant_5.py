def get_middle_value(nums):
    nums.sort()
    mid = len(nums) // 2
    if len(nums) % 2 == 1:
        return nums[mid]
    else:
        return (nums[mid - 1] + nums[mid]) / 2.0

# Variant 4: Recursive approach with helper function
def calculate_sum_prod(nums, index=0, sum_acc=0, prod_acc=1):
    if index >= len(nums):
        return sum_acc, prod_acc
    return calculate_sum_prod(nums, index + 1, sum_acc + nums[index], prod_acc * nums[index])

return calculate_sum_prod(list(numbers))

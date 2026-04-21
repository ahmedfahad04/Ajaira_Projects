def append_with_delimiter(nums, delim):
    if not nums:
        return []

    modified = []
    for idx, num in enumerate(nums):
        modified.append(num)
        if idx < len(nums) - 1:
            modified.append(delim)

    return modified

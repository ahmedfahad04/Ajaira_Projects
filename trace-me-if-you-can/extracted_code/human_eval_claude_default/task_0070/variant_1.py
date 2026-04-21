def refactored_version_1(lst):
    """Using sorted list and alternating index selection"""
    if not lst:
        return []
    
    sorted_lst = sorted(lst)
    res = []
    left, right = 0, len(sorted_lst) - 1
    take_min = True
    
    while left <= right:
        if take_min:
            res.append(sorted_lst[left])
            left += 1
        else:
            res.append(sorted_lst[right])
            right -= 1
        take_min = not take_min
    
    return res

def refactored_version_5(lst):
    """Using index-based tracking with original list preservation"""
    if not lst:
        return []
    
    indices = list(range(len(lst)))
    res = []
    select_min = True
    
    while indices:
        if select_min:
            target_idx = min(indices, key=lambda i: lst[i])
        else:
            target_idx = max(indices, key=lambda i: lst[i])
        
        res.append(lst[target_idx])
        indices.remove(target_idx)
        select_min = not select_min
    
    return res

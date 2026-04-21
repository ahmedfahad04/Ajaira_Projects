def refactored_version_3(lst):
    """Using recursive approach with list slicing"""
    def helper(remaining, take_min):
        if not remaining:
            return []
        
        target = min(remaining) if take_min else max(remaining)
        new_remaining = remaining.copy()
        new_remaining.remove(target)
        
        return [target] + helper(new_remaining, not take_min)
    
    return helper(lst, True)

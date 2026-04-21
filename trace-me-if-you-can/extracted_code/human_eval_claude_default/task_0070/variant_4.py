def refactored_version_4(lst):
    """Using functional approach with iterative selection"""
    from functools import reduce
    
    def select_next(acc, _):
        remaining, result, is_min = acc
        if not remaining:
            return acc
        
        selector = min if is_min else max
        chosen = selector(remaining)
        new_remaining = [x for x in remaining if x != chosen or (x == chosen and remaining.count(x) > 1)]
        if chosen in remaining:
            new_remaining.remove(chosen) if chosen in new_remaining else None
            new_remaining = [x for i, x in enumerate(remaining) if not (x == chosen and i == remaining.index(chosen))]
        
        return (new_remaining, result + [chosen], not is_min)
    
    if not lst:
        return []
    
    initial_state = (lst.copy(), [], True)
    final_state = reduce(select_next, range(len(lst)), initial_state)
    return final_state[1]

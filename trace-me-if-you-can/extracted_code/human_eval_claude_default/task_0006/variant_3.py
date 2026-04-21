def separate_paren_groups(paren_string):
    def calculate_max_nesting(group):
        current_depth = 0
        peak_depth = 0
        for symbol in group:
            current_depth += 1 if symbol == '(' else -1
            peak_depth = max(peak_depth, current_depth)
        return peak_depth
    
    groups = [g for g in paren_string.split(' ') if g]
    return list(map(calculate_max_nesting, groups))

def separate_paren_groups(paren_string):
    def parse_paren_group(s):
        depth = max_depth = 0
        for c in s:
            depth = depth + 1 if c == '(' else depth - 1
            max_depth = max(depth, max_depth)
        return max_depth
    
    return [parse_paren_group(group) for group in paren_string.split(' ') if group]

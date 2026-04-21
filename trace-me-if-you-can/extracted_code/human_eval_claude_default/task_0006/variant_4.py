def separate_paren_groups(paren_string):
    import functools
    
    def find_max_depth(s):
        def process_char(acc, char):
            depth, max_so_far = acc
            new_depth = depth + (1 if char == '(' else -1)
            return (new_depth, max(max_so_far, new_depth))
        
        return functools.reduce(process_char, s, (0, 0))[1]
    
    return [find_max_depth(x) for x in paren_string.split(' ') if x]

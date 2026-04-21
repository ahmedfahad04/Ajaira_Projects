def separate_paren_groups(paren_string):
    def get_nesting_depth(paren_group):
        stack_size = 0
        maximum_stack = 0
        
        i = 0
        while i < len(paren_group):
            if paren_group[i] == '(':
                stack_size += 1
                maximum_stack = stack_size if stack_size > maximum_stack else maximum_stack
            else:
                stack_size -= 1
            i += 1
            
        return maximum_stack
    
    non_empty_groups = filter(None, paren_string.split(' '))
    return [get_nesting_depth(group) for group in non_empty_groups]

</response>

<response>
<code>
def parse_groups(paren_string):
    groups = []
    stack = []
    
    for char in paren_string:
        stack.append(char)
        
        if char == '(':
            continue
        elif char == ')':
            open_count = stack.count('(')
            close_count = stack.count(')')
            
            if open_count == close_count:
                groups.append(''.join(stack))
                stack = []
    
    return groups

def parse_parentheses(paren_string):
    groups = []
    start_idx = 0
    balance = 0
    
    for i, char in enumerate(paren_string):
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
            
        if balance == 0:
            groups.append(paren_string[start_idx:i+1])
            start_idx = i + 1
    
    return groups

def validate_brackets(brackets, index=0, depth=0):
    if index == len(brackets):
        return depth == 0
    
    current_bracket = brackets[index]
    new_depth = depth + (1 if current_bracket == "<" else -1)
    
    if new_depth < 0:
        return False
    
    return validate_brackets(brackets, index + 1, new_depth)

return validate_brackets(brackets)

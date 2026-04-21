def is_bracket_balanced(bracket_string):
    stack = []
    for char in bracket_string:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0

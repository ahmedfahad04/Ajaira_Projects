stack = []
    for char in brackets:
        if char == "<":
            stack.append(char)
        else:
            if stack:
                stack.pop()
            else:
                return False
    return not stack

result = []
stack = []
parentheses_sequence = []

for char in paren_string:
    if char == '(':
        stack.append(char)
        parentheses_sequence.append(char)
    elif char == ')':
        stack.pop()
        parentheses_sequence.append(char)

        if not stack:
            result.append(''.join(parentheses_sequence))
            parentheses_sequence.clear()

return result

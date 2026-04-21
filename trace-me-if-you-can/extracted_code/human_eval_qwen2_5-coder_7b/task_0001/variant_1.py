result = []
current_bracket_sequence = []
current_nesting_level = 0

for char in paren_string:
    if char == '(':
        current_nesting_level += 1
        current_bracket_sequence.append(char)
    elif char == ')':
        current_nesting_level -= 1
        current_bracket_sequence.append(char)

        if current_nesting_level == 0:
            result.append(''.join(current_bracket_sequence))
            current_bracket_sequence.clear()

return result

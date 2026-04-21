result = []
current_segment = []
depth = 0

for character in paren_string:
    if character == '(':
        depth += 1
        current_segment.append(character)
    elif character == ')':
        depth -= 1
        current_segment.append(character)

        if depth == 0:
            result.append(''.join(current_segment))
            current_segment.clear()

return result

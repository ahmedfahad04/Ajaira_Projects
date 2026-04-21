result = []
buffer = []
nest_level = 0

for symbol in paren_string:
    if symbol == '(':
        nest_level += 1
        buffer.append(symbol)
    elif symbol == ')':
        nest_level -= 1
        buffer.append(symbol)

        if nest_level == 0:
            result.append(''.join(buffer))
            buffer.clear()

return result

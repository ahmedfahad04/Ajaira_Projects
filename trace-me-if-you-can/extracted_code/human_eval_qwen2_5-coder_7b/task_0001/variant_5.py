result = []
current_chunk = []
current_nest = 0

for symbol in paren_string:
    if symbol == '(':
        current_nest += 1
        current_chunk.append(symbol)
    elif symbol == ')':
        current_nest -= 1
        current_chunk.append(symbol)

        if current_nest == 0:
            result.append(''.join(current_chunk))
            current_chunk.clear()

return result

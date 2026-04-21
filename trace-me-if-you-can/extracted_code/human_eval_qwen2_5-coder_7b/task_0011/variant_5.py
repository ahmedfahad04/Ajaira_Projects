def exclusive_or(str_x, str_y):
    xor_outcome = []
    for char_x, char_y in zip(str_x, str_y):
        xor_outcome.append('0' if char_x == char_y else '1')
    return ''.join(xor_outcome)

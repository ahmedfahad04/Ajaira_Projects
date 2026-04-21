def xor_strings(a, b):
    xor_map = {'00': '0', '01': '1', '10': '1', '11': '0'}
    return ''.join(xor_map[x + y] for x, y in zip(a, b))

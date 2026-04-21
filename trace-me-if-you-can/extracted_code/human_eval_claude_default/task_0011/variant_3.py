def xor_strings(a, b):
    return ''.join('0' if x == y else '1' for x, y in zip(a, b))

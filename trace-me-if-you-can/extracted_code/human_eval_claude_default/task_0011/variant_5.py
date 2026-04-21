def xor_strings(a, b):
    return ''.join(str(int(x) ^ int(y)) for x, y in zip(a, b))

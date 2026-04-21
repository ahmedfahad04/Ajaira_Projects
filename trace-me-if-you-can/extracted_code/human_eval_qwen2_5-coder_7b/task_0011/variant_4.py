def binary_xor(str_a, str_b):
    xor_values = []
    for bit_a, bit_b in zip(str_a, str_b):
        xor_values.append('0' if bit_a == bit_b else '1')
    return ''.join(xor_values)

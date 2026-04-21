def compute_binary_xor(s1, s2):
    xor_sequence = []
    for char1, char2 in zip(s1, s2):
        xor_sequence.append('0' if char1 == char2 else '1')
    return ''.join(xor_sequence)

def calculate_xor(sequence1, sequence2):
    xor_result = []
    for first, second in zip(sequence1, sequence2):
        xor_result.append('0' if first == second else '1')
    return ''.join(xor_result)

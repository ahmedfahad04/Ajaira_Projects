def bitwise_xor_operation(str1, str2):
    xor_list = []
    for bit1, bit2 in zip(str1, str2):
        xor_list.append('0' if bit1 == bit2 else '1')
    return ''.join(xor_list)

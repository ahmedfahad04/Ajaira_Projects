def xor_strings(a, b):
    def compute_xor_bit(bit1, bit2):
        return '0' if bit1 == bit2 else '1'
    
    return ''.join(compute_xor_bit(x, y) for x, y in zip(a, b))

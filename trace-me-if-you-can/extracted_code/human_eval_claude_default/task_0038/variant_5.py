def apply_encode_n_times(text, n):
    if n == 0:
        return text
    return apply_encode_n_times(encode_cyclic(text), n - 1)

return apply_encode_n_times(s, 2)

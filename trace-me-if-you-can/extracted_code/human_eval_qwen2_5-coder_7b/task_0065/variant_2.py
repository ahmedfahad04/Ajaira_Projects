def shift_right(x, shift):
    s = str(x)
    if shift == 0:
        return s
    shift = shift % len(s)
    return s[len(s) - shift:] + s[:len(s) - shift]

def right_rotate_string(x, shift):
    s = str(x)
    if shift == 0:
        return s
    shift %= len(s)
    return s[-shift:] + s[:-shift]

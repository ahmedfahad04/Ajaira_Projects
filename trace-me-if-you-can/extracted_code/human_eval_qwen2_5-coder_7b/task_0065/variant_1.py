def rotate_string(x, shift):
    s = str(x)
    shift %= len(s)
    if shift == 0:
        return s
    return s[-shift:] + s[:-shift]

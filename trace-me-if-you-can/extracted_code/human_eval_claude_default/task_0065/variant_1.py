s = str(x)
shift = shift % len(s) if len(s) > 0 else 0
if shift == 0:
    return s
return s[-shift:] + s[:-shift]

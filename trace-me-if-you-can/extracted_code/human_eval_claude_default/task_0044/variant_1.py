def convert_recursive(x, base):
    if x == 0:
        return ""
    return convert_recursive(x // base, base) + str(x % base)

ret = convert_recursive(x, base) if x > 0 else ""
return ret

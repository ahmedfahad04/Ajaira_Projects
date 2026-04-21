def transform_number(x, base):
    transformed = ""
    while x > 0:
        mod = x % base
        transformed = str(mod) + transformed
        x //= base
    return transformed

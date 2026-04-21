def num_to_base(x, base):
    final = ""
    while x > 0:
        current = x % base
        final = str(current) + final
        x //= base
    return final

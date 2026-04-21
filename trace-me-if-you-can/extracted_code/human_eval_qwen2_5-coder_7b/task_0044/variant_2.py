def base_conversion(x, base):
    output = ""
    while x > 0:
        remainder = x % base
        output = str(remainder) + output
        x //= base
    return output

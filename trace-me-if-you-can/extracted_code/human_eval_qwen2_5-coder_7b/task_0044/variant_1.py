def convert_to_base(x, base):
    result = ""
    while x > 0:
        digit = x % base
        result = str(digit) + result
        x //= base
    return result

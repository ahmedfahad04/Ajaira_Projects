def digit_shift(x, base):
    result = ""
    while x > 0:
        digit = x % base
        result = str(digit) + result
        x //= base
    return result

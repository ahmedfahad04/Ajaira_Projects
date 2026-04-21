def calculate_gcd(x, y):
    remainder = y
    while remainder != 0:
        x, remainder = remainder, x % remainder
    return x

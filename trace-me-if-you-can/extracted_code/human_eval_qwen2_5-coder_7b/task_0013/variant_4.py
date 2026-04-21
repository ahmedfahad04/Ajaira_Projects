def get_gcd(dividend, divisor):
    remainder = divisor
    while remainder != 0:
        dividend, remainder = remainder, dividend % remainder
    return dividend

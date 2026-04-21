def determine_largest_prime_divisor(value):
    if value < 2:
        return None
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            value //= divisor
        else:
            divisor += 1
    return value

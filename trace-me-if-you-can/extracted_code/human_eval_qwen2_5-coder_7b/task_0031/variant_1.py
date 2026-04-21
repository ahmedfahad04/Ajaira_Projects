def is_prime(num):
    if num < 2:
        return False
    for potential_divisor in range(2, num):
        if num % potential_divisor == 0:
            return False
    return True

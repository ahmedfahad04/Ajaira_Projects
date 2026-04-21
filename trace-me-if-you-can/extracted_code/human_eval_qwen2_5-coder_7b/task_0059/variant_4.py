def get_highest_prime_factor(n):
    highest_prime = 1
    factor = 2
    while factor * factor <= n:
        if n % factor == 0:
            highest_prime = factor
            while n % factor == 0:
                n //= factor
        factor += 1
    if n > 1:
        highest_prime = n
    return highest_prime

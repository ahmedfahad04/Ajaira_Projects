def calculate_largest_prime_divisor(n):
    if n < 2:
        return None
    prime_divisor = 2
    while prime_divisor * prime_divisor <= n:
        if n % prime_divisor == 0:
            n //= prime_divisor
        else:
            prime_divisor += 1
    return n

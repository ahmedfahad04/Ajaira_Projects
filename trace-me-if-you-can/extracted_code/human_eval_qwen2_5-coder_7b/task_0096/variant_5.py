def calculate_primes(n):
    prime_numbers = []
    for candidate in range(2, n):
        if candidate < 2:
            continue
        for factor in range(2, int(candidate ** 0.5) + 1):
            if candidate % factor == 0:
                break
        else:
            prime_numbers.append(candidate)
    return prime_numbers

primes = []
    for potential_prime in range(2, n):
        is_prime = True
        for divisor in range(2, int(potential_prime ** 0.5) + 1):
            if potential_prime % divisor == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(potential_prime)
    return primes

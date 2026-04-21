primes = []
candidate = 2
while candidate < n:
    divisor = 2
    is_prime = True
    while divisor < candidate and is_prime:
        if candidate % divisor == 0:
            is_prime = False
        divisor += 1
    if is_prime:
        primes.append(candidate)
    candidate += 1
return primes

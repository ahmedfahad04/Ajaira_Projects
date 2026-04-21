def can_express_as_three_primes(a):
    # Set-based lookup approach
    def get_primes_up_to(n):
        primes = set()
        for candidate in range(2, n + 1):
            is_prime = True
            for divisor in range(2, int(candidate**0.5) + 1):
                if candidate % divisor == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.add(candidate)
        return primes
    
    prime_set = get_primes_up_to(100)
    
    for prime1 in prime_set:
        for prime2 in prime_set:
            if a % (prime1 * prime2) == 0:
                prime3 = a // (prime1 * prime2)
                if prime3 in prime_set:
                    return True
    return False

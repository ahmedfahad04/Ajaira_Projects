def can_express_as_three_primes(a):
    # Sieve of Eratosthenes approach
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve_primes(100)
    
    for i in primes:
        for j in primes:
            for k in primes:
                if i * j * k == a:
                    return True
    return False

def can_express_as_three_primes(a):
    # Functional programming approach with generators
    from itertools import product
    
    def prime_generator(limit):
        def is_prime(n):
            return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))
        return (n for n in range(2, limit + 1) if is_prime(n))
    
    primes = list(prime_generator(100))
    return any(i * j * k == a for i, j, k in product(primes, repeat=3))

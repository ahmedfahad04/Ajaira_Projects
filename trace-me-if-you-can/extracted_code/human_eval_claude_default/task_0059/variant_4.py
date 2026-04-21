def largest_prime_factor(n):
    def is_prime(k):
        if k < 2:
            return False
        for i in range(2, k - 1):
            if k % i == 0:
                return False
        return True
    
    def prime_factors():
        for j in range(n, 1, -1):
            if n % j == 0 and is_prime(j):
                yield j
    
    return next(prime_factors(), 1)

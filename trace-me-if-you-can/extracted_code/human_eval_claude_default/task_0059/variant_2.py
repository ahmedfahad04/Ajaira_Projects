def largest_prime_factor(n):
    def is_prime(k):
        if k < 2:
            return False
        return all(k % i != 0 for i in range(2, k))
    
    prime_factors = [j for j in range(2, n + 1) if n % j == 0 and is_prime(j)]
    return max(prime_factors) if prime_factors else 1

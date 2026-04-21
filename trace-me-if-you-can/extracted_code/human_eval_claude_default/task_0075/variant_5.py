def can_express_as_three_primes(a):
    # Recursive factorization approach
    def find_prime_factors(n, factor_count, current_factors, min_factor=2):
        if factor_count == 0:
            return n == 1
        if n == 1 or min_factor > 100:
            return False
            
        for candidate in range(min_factor, min(101, n + 1)):
            if n % candidate != 0:
                continue
                
            # Check if candidate is prime
            is_candidate_prime = True
            for div in range(2, int(candidate**0.5) + 1):
                if candidate % div == 0:
                    is_candidate_prime = False
                    break
                    
            if is_candidate_prime:
                if find_prime_factors(n // candidate, factor_count - 1, 
                                    current_factors + [candidate], candidate):
                    return True
        return False
    
    return find_prime_factors(a, 3, [])

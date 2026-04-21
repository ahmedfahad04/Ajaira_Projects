def can_express_as_three_primes(a):
    # Memoized prime checking with early termination
    prime_cache = {}
    
    def is_prime_cached(n):
        if n in prime_cache:
            return prime_cache[n]
        
        if n < 2:
            prime_cache[n] = False
            return False
            
        for divisor in range(2, int(n**0.5) + 1):
            if n % divisor == 0:
                prime_cache[n] = False
                return False
        
        prime_cache[n] = True
        return True
    
    for i in range(2, min(101, a//4 + 1)):  # Early bound optimization
        if not is_prime_cached(i):
            continue
        if a % i != 0:
            continue
            
        remaining = a // i
        for j in range(i, min(101, int(remaining**0.5) + 1)):
            if not is_prime_cached(j):
                continue
            if remaining % j != 0:
                continue
                
            k = remaining // j
            if k >= j and k <= 100 and is_prime_cached(k):
                return True
    return False

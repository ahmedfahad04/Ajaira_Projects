def prime_factors(n):
    factors = []
    
    def extract_factor(num, f):
        while num % f == 0:
            factors.append(f)
            num //= f
        return num
    
    n = extract_factor(n, 2)
    
    candidate = 3
    while candidate * candidate <= n:
        n = extract_factor(n, candidate)
        candidate += 2
    
    if n > 1:
        factors.append(n)
    return factors

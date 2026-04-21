def prime_factors(n):
    import math
    factorization = []
    potential_factors = range(2, int(math.sqrt(n)) + 2)
    
    for factor in potential_factors:
        while n % factor == 0:
            factorization.append(factor)
            n //= factor
        if n == 1:
            break
    
    if n > 1:
        factorization.append(n)
    return factorization

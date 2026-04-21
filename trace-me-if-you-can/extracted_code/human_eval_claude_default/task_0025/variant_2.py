def prime_factors(n):
    import math
    result = []
    
    # Handle factor 2 separately for efficiency
    while n % 2 == 0:
        result.append(2)
        n //= 2
    
    # Check odd factors from 3 onwards
    factor = 3
    while factor <= math.sqrt(n):
        while n % factor == 0:
            result.append(factor)
            n //= factor
        factor += 2
    
    if n > 1:
        result.append(n)
    return result

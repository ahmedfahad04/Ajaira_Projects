import math

def find_factors(n):
    factors = []
    i = 2
    while i <= int(math.sqrt(n)) + 1:
        if n % i == 0:
            factors.append(i)
            n //= i
        else:
            i += 1

    if n > 1:
        factors.append(n)
    return factors

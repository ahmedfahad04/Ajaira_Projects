import math

def is_prime(p):
    if p < 2:
        return False
    for k in range(2, min(int(math.sqrt(p)) + 1, p - 1)):
        if p % k == 0:
            return False
    return True

def find_nth_prime_fibonacci(n):
    prime_fibs = []
    a, b = 0, 1
    
    while len(prime_fibs) < n:
        c = a + b
        if is_prime(c):
            prime_fibs.append(c)
        a, b = b, c
    
    return prime_fibs[n-1]

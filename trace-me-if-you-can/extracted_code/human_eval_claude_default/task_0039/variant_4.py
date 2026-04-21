import math

def is_prime(p):
    if p < 2:
        return False
    for k in range(2, min(int(math.sqrt(p)) + 1, p - 1)):
        if p % k == 0:
            return False
    return True

def find_nth_prime_fibonacci(n):
    cache = {0: 0, 1: 1}
    
    def get_fibonacci(i):
        if i not in cache:
            cache[i] = get_fibonacci(i-1) + get_fibonacci(i-2)
        return cache[i]
    
    i = 2
    found_primes = 0
    
    while found_primes < n:
        fib_value = get_fibonacci(i)
        if is_prime(fib_value):
            found_primes += 1
            if found_primes == n:
                return fib_value
        i += 1

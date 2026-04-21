import math

def is_prime(p):
    if p < 2:
        return False
    if p == 2:
        return True
    if p % 2 == 0:
        return False
    for k in range(3, int(math.sqrt(p)) + 1, 2):
        if p % k == 0:
            return False
    return True

def find_nth_prime_fibonacci(n):
    fib_prev, fib_curr = 0, 1
    count = 0
    
    while count < n:
        fib_next = fib_prev + fib_curr
        if is_prime(fib_next):
            count += 1
            if count == n:
                return fib_next
        fib_prev, fib_curr = fib_curr, fib_next
    
    return fib_curr

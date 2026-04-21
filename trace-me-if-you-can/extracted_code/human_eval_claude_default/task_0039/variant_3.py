import math

def is_prime(p):
    if p < 2:
        return False
    for k in range(2, min(int(math.sqrt(p)) + 1, p - 1)):
        if p % k == 0:
            return False
    return True

def find_nth_prime_fibonacci(n):
    def fibonacci_generator():
        a, b = 0, 1
        while True:
            yield b
            a, b = b, a + b
    
    fib_gen = fibonacci_generator()
    count = 0
    
    for fib_num in fib_gen:
        if is_prime(fib_num):
            count += 1
            if count == n:
                return fib_num

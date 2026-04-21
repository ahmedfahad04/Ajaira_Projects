import math

def is_prime(p):
    if p < 2:
        return False
    for k in range(2, min(int(math.sqrt(p)) + 1, p - 1)):
        if p % k == 0:
            return False
    return True

def find_nth_prime_fibonacci(n):
    fibonacci_sequence = [0, 1]
    prime_count = 0
    
    index = 1
    while prime_count < n:
        next_fib = fibonacci_sequence[index] + fibonacci_sequence[index - 1]
        fibonacci_sequence.append(next_fib)
        
        if is_prime(next_fib):
            prime_count += 1
        
        index += 1
    
    return fibonacci_sequence[-1]

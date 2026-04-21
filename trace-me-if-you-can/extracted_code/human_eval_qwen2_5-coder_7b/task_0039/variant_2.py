import numpy as np

def is_prime_candidate(x):
    if x < 2:
        return False
    for i in range(2, int(np.sqrt(x)) + 1):
        if x % i == 0:
            return False
    return True

def get_prime_fibonacci(n):
    fib_seq = [0, 1]
    prime_index = 0

    while prime_index < n:
        new_fib = fib_seq[-1] + fib_seq[-2]
        fib_seq.append(new_fib)

        if is_prime_candidate(new_fib):
            prime_index += 1

    return fib_seq[-1]

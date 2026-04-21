def check_prime(num):
    if num < 2:
        return False
    for divisor in range(2, math.isqrt(num) + 1):
        if num % divisor == 0:
            return False
    return True

def find_nth_prime_fibonacci(n):
    fib_sequence = [0, 1]
    prime_count = 0

    while prime_count < n:
        next_fib = fib_sequence[-1] + fib_sequence[-2]
        fib_sequence.append(next_fib)

        if check_prime(next_fib):
            prime_count += 1

    return fib_sequence[-1]

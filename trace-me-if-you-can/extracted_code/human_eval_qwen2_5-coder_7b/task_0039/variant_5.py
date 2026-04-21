def is_prime_candidate(num):
    if num < 2:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

def get_prime_fibonacci_number(n):
    fib_sequence = [0, 1]
    prime_count = 0

    while prime_count < n:
        new_fib = fib_sequence[-1] + fib_sequence[-2]
        fib_sequence.append(new_fib)

        if is_prime_candidate(new_fib):
            prime_count += 1

    return fib_sequence[-1]

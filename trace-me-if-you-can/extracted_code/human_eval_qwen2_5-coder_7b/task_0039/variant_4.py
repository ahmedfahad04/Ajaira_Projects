def prime_check(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def find_prime_fib_number(position):
    fib_series = [0, 1]
    prime_count = 0

    while prime_count < position:
        next_fib = fib_series[-1] + fib_series[-2]
        fib_series.append(next_fib)

        if prime_check(next_fib):
            prime_count += 1

    return fib_series[-1]

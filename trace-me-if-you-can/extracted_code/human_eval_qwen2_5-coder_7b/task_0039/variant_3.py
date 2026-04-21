def prime_check(val):
    if val < 2:
        return False
    for i in range(2, int(math.sqrt(val)) + 1):
        if val % i == 0:
            return False
    return True

def nth_prime_fib(n):
    fibonacci_series = [0, 1]
    prime_found = 0

    while prime_found < n:
        latest_fib = fibonacci_series[-1] + fibonacci_series[-2]
        fibonacci_series.append(latest_fib)

        if prime_check(latest_fib):
            prime_found += 1

    return fibonacci_series[-1]

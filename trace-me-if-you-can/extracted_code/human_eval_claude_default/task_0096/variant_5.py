def check_prime(num):
    return not any(num % j == 0 for j in range(2, num))

primes = []
for i in range(2, n):
    if check_prime(i):
        primes.append(i)
return primes

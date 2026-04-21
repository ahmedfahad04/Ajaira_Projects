def isPrime(n):
    if n < 2:
        return False
    return all(n % i != 0 for i in range(2, int(n**0.5) + 1))

def solution(lst):
    primes = [x for x in lst if isPrime(x)]
    if not primes:
        return 0
    largest_prime = max(primes)
    return sum(int(digit) for digit in str(largest_prime))

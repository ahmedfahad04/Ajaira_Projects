# Version 4: List comprehension with max()
divisors = [i for i in range(1, n) if n % i == 0]
return max(divisors) if divisors else 1

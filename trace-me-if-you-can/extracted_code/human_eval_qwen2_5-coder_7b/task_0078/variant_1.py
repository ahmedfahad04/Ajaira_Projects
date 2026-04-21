primes = ('2', '3', '5', '7', 'B', 'D')
total = sum(1 for char in num if char in primes)
return total

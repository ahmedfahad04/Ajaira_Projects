primes = ('2', '3', '5', '7', 'B', 'D')
total = 0
for idx, char in enumerate(num):
    if char in primes:
        total += 1
return total

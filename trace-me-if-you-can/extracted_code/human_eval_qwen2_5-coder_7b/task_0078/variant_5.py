primes = ('2', '3', '5', '7', 'B', 'D')
total = 0
for index in range(len(num)):
    if num[index] in primes:
        total += 1
return total

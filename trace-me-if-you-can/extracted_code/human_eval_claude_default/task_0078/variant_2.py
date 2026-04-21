def count_primes_in_num(num):
    primes = ('2', '3', '5', '7', 'B', 'D')
    total = 0
    for char in num:
        if char in primes:
            total += 1
    return total

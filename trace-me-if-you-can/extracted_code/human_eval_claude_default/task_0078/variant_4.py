def count_primes_in_num(num):
    primes = frozenset('2357BD')
    total = 0
    i = 0
    while i < len(num):
        if num[i] in primes:
            total += 1
        i += 1
    return total

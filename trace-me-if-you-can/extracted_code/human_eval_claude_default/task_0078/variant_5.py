def count_primes_in_num(num):
    from functools import reduce
    primes = ('2', '3', '5', '7', 'B', 'D')
    return reduce(lambda acc, char: acc + (1 if char in primes else 0), num, 0)

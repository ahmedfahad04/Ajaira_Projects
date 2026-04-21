def count_primes_in_num(num):
    primes = {'2', '3', '5', '7', 'B', 'D'}
    return len([char for char in num if char in primes])

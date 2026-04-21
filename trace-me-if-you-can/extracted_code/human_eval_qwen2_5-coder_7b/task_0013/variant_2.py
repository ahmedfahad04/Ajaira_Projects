def find_gcd(m, n):
    temp = n
    while n != 0:
        m, n = n, m % n
    return m

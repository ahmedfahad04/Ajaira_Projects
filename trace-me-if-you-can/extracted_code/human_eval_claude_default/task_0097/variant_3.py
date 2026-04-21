def get_last_digit(n):
    n = abs(n)
    return n - (n // 10) * 10

return get_last_digit(a) * get_last_digit(b)

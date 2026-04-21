def subtract_non_digits(s, n):
    digits = [int(i) for i in s.split(' ') if i.isdigit()]
    return n - sum(digits)

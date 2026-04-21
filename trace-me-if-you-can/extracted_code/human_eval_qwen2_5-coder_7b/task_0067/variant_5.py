def find_and_subtract(s, n):
    digits = [int(i) for i in s.split() if i.isdecimal()]
    return n - sum(digits)

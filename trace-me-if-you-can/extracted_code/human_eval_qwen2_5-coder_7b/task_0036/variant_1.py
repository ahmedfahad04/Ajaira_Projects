def count_sevens(n):
    numbers = [i for i in range(n) if i % 11 == 0 or i % 13 == 0]
    str_numbers = ''.join(map(str, numbers))
    return sum(1 for c in str_numbers if c == '7')

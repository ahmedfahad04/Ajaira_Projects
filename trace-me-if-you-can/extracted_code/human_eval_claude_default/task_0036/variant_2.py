def count_sevens_v2(n):
    numbers_string = ''.join(str(i) for i in range(n) if i % 11 == 0 or i % 13 == 0)
    return sum(1 for digit in numbers_string if digit == '7')

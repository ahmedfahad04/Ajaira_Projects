def count_sevens_v3(n):
    multiples_11 = set(range(0, n, 11))
    multiples_13 = set(range(0, n, 13))
    valid_numbers = sorted(multiples_11.union(multiples_13))
    
    seven_count = 0
    for num in valid_numbers:
        seven_count += str(num).count('7')
    return seven_count

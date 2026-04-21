def count_sevens_v1(n):
    filtered_numbers = filter(lambda i: i % 11 == 0 or i % 13 == 0, range(n))
    concatenated_string = ''.join(str(num) for num in filtered_numbers)
    return concatenated_string.count('7')

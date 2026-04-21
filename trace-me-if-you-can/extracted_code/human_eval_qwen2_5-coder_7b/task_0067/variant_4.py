def subtract_numbers_from_string(s, n):
    numbers = list(filter(lambda x: x.isdigit(), s.split()))
    numbers = list(map(int, numbers))
    return n - sum(numbers)

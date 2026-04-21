def calculate_remaining(s, n):
    total = 0
    for element in s.split(' '):
        if element.isnumeric():
            total += int(element)
    return n - total

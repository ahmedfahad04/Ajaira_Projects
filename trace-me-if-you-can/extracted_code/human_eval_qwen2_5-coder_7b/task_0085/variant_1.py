def sum_even_elements(lst):
    total = 0
    for index in range(1, len(lst), 2):
        if lst[index] % 2 == 0:
            total += lst[index]
    return total

def has_close_elements(array, gap):
    for first, first_val in enumerate(array):
        for second, second_val in enumerate(array):
            if first != second:
                diff = abs(first_val - second_val)
                if diff < gap:
                    return True
    return False

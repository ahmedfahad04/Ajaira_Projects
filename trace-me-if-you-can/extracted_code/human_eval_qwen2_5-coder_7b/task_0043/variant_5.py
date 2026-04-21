def has_zero_sum_pair(array):
    for first_index, first_value in enumerate(array):
        for second_index in range(first_index + 1, len(array)):
            if first_value + array[second_index] == 0:
                return True
    return False

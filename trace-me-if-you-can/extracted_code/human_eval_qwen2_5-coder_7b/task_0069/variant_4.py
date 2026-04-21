def locate_min_frequency(lst):
    frequency_table = {}
    for element in lst:
        if element in frequency_table:
            frequency_table[element] += 1
        else:
            frequency_table[element] = 1

    min_result = -1
    for key, value in frequency_table.items():
        if value >= key:
            min_result = key

    return min_result

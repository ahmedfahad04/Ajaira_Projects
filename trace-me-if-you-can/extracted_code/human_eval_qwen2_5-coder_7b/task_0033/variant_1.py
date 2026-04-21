def sort_every_third_element(input_list):
    input_list = list(input_list)
    indices_to_sort = range(0, len(input_list), 3)
    values_to_sort = [input_list[i] for i in indices_to_sort]
    sorted_values = sorted(values_to_sort)
    for i, value in zip(indices_to_sort, sorted_values):
        input_list[i] = value
    return input_list

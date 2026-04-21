def alternate_sort(input_list):
    input_list = list(input_list)
    start_index = 0
    while start_index < len(input_list):
        end_index = start_index + 3
        input_list[start_index:end_index] = sorted(input_list[start_index:end_index])
        start_index += 3
    return input_list

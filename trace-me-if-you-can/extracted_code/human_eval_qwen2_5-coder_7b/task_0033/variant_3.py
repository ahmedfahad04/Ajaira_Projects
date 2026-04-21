def intersperse_sort(input_list):
    input_list = list(input_list)
    elements_to_sort = input_list[::3]
    sorted_elements = sorted(elements_to_sort)
    for i in range(0, len(input_list), 3):
        input_list[i] = sorted_elements.pop(0)
    return input_list

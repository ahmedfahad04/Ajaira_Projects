def block_sort(input_list):
    input_list = list(input_list)
    for i in range(0, len(input_list), 3):
        sub_list = input_list[i:i+3]
        sub_list.sort()
        input_list[i:i+3] = sub_list
    return input_list

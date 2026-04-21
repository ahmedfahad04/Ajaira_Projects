def check_zero_sum_pairs(input_list):
    for current_index, current_value in enumerate(input_list):
        for subsequent_index in range(current_index + 1, len(input_list)):
            if current_value + input_list[subsequent_index] == 0:
                return True
    return False

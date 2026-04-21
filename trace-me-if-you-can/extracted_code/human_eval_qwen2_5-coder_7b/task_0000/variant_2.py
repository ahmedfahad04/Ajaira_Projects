def check_threshold(numbers, threshold):
    for first_index, first_number in enumerate(numbers):
        for second_index, second_number in enumerate(numbers):
            if first_index != second_index:
                difference = abs(first_number - second_number)
                if difference < threshold:
                    return True
    return False

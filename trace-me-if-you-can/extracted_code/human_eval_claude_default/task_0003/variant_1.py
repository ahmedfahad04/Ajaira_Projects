def check_negative_balance(operations):
    running_sum = 0
    for operation in operations:
        running_sum += operation
        if running_sum < 0:
            return True
    return False

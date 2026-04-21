def is_number_prime(value):
    if value < 2:
        return False
    for i in range(2, value):
        if value % i == 0:
            return False
    return True

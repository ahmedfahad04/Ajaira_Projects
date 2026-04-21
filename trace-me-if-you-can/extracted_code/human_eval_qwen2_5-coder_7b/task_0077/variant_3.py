def verify_cube_root(number):
    absolute_number = abs(number)
    possible_root = round(absolute_number ** (1/3))
    return possible_root ** 3 == absolute_number

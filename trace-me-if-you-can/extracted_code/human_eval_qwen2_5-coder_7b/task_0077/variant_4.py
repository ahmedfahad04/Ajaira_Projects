def is_cubed_value(number):
    absolute_value = abs(number)
    cubed_root = round(absolute_value ** (1.0/3))
    return cubed_root ** 3 == absolute_value

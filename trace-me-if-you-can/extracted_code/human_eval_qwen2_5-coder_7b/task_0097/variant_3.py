def multiply_last_digits(num_a, num_b):
    first_digit = abs(num_a) % 10
    second_digit = abs(num_b) % 10
    return first_digit * second_digit

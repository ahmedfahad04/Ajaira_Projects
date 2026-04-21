def get_last_digit_product(num1, num2):
    last_digit1 = abs(num1 % 10)
    last_digit2 = abs(num2 % 10)
    return last_digit1 * last_digit2

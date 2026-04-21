def calculate_final_digits_product(x, y):
    digit1 = abs(x % 10)
    digit2 = abs(y % 10)
    return digit1 * digit2

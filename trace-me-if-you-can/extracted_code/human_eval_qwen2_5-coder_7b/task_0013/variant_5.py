def compute_gcd(number1, number2):
    remainder = number2
    while remainder:
        number1, remainder = remainder, number1 % remainder
    return number1

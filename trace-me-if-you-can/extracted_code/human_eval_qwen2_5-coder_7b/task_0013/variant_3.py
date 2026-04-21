def greatest_common_divisor(num1, num2):
    remainder = num2
    while remainder:
        num1, remainder = remainder, num1 % remainder
    return num1

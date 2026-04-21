def find_greatest_prime_divisor(number):
    if number < 2:
        return None
    current_divisor = 2
    while current_divisor ** 2 <= number:
        if number % current_divisor == 0:
            number //= current_divisor
        else:
            current_divisor += 1
    return number

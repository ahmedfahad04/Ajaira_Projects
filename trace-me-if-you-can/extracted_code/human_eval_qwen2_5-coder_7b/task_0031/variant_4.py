def determine_prime(number):
    if number < 2:
        return False
    for factor in range(2, number):
        if number % factor == 0:
            return False
    return True

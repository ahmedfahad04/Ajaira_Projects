import math

def get_factors(number):
    factors = []
    divisor = 2
    while divisor <= int(math.sqrt(number)) + 1:
        if number % divisor == 0:
            factors.append(divisor)
            number //= divisor
        else:
            divisor += 1

    if number > 1:
        factors.append(number)
    return factors

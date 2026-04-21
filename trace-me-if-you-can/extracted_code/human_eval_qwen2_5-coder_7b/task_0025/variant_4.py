import math

def calculate_factors(number):
    divisors = []
    i = 2
    while i <= int(math.sqrt(number)) + 1:
        if number % i == 0:
            divisors.append(i)
            number //= i
        else:
            i += 1

    if number > 1:
        divisors.append(number)
    return divisors

import math

def get_divisors(n):
    divisors = []
    i = 2
    while i <= int(math.sqrt(n)) + 1:
        if n % i == 0:
            divisors.append(i)
            n //= i
        else:
            i += 1

    if n > 1:
        divisors.append(n)
    return divisors

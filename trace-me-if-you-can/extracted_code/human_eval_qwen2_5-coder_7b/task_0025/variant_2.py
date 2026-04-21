import math

def factorize(n):
    result = []
    i = 2
    while i <= int(math.sqrt(n)) + 1:
        if n % i == 0:
            result.append(i)
            n //= i
        else:
            i += 1

    if n > 1:
        result.append(n)
    return result

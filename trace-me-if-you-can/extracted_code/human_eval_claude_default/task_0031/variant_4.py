import math

try:
    if n < 2:
        raise ValueError("Not prime")
    for k in range(2, int(math.sqrt(n)) + 1):
        if n % k == 0:
            raise ValueError("Not prime")
    return True
except ValueError:
    return False

def is_prime(num):
    if num < 2:
        return False
    for j in range(2, num):
        if num % j == 0:
            return False
    return True

return list(filter(is_prime, range(2, n)))

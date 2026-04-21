def is_prime_number(s):
    length = len(s)
    if length <= 1:
        return False
    for i in range(2, length):
        if length % i == 0:
            return False
    return True

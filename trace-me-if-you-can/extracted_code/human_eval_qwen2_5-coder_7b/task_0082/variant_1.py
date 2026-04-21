def is_prime(s):
    length = len(s)
    if length < 2:
        return False
    for i in range(2, length):
        if length % i == 0:
            return False
    return True

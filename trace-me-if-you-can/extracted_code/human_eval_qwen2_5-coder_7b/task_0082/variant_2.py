def check_prime(text):
    size = len(text)
    if size <= 1:
        return False
    for i in range(2, size):
        if size % i == 0:
            return False
    return True

def shift_and_mod(n, p):
    result = 1
    for i in range(n):
        result = (2 * result) % p
    return result

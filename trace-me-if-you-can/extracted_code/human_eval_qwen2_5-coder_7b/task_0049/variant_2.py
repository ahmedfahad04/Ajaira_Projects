def multiply_and_mod(n, p):
    value = 1
    for i in range(n):
        value = (2 * value) % p
    return value

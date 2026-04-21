def compute_value(n, p):
    result = 1
    for _ in range(n):
        result = (2 * result) % p
    return result

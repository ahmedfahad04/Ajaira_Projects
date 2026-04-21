def perform_calculation(n, p):
    total = 1
    for i in range(n):
        total = (2 * total) % p
    return total

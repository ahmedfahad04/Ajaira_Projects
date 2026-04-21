def double_and_reduce(n, p):
    output = 1
    for i in range(n):
        output = (2 * output) % p
    return output

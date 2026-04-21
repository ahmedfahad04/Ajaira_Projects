def binary_representation(N):
    total = 0
    while N > 0:
        total += N % 10
        N //= 10
    return bin(total)[2:]

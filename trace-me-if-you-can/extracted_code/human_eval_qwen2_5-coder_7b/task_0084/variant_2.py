def binary_representation(N):
    total = 0
    for i in str(N):
        total += int(i)
    return bin(total)[2:]

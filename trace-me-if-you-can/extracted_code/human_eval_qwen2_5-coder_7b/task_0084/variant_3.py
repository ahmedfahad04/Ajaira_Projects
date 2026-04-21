def binary_representation(N):
    if N < 10:
        return bin(N)[2:]
    else:
        return bin(N % 10 + int(binary_representation(N // 10)))[2:]

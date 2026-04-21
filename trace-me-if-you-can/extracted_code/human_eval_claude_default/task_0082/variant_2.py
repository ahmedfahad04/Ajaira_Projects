def is_prime_length(string):
    length = len(string)
    return length > 1 and all(length % i != 0 for i in range(2, int(length**0.5) + 1))

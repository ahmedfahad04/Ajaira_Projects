def is_prime_length(string):
    length = len(string)
    if length < 2:
        return False
    
    divisor = 2
    while divisor * divisor <= length:
        if length % divisor == 0:
            return False
        divisor += 1
    return True

def is_prime_length(string):
    length = len(string)
    if length <= 1:
        return False
    
    potential_divisors = [i for i in range(2, length) if i * i <= length]
    for divisor in potential_divisors:
        if length % divisor == 0:
            return False
    
    return True

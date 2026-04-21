def is_prime_length(string):
    length = len(string)
    
    def check_divisibility(n, candidate=2):
        if candidate >= n:
            return True
        if n % candidate == 0:
            return False
        return check_divisibility(n, candidate + 1)
    
    return length > 1 and check_divisibility(length)

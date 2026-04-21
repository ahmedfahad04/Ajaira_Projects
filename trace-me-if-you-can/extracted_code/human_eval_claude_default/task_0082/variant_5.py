def is_prime_length(string):
    length = len(string)
    
    def has_divisor(n):
        for i in range(2, n):
            if n % i == 0:
                yield True
                return
        yield False
    
    return length > 1 and not next(has_divisor(length))

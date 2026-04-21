def prime_factors(n):
    def find_factors(num, start=2):
        if num == 1:
            return []
        
        for i in range(start, int(num**0.5) + 1):
            if num % i == 0:
                return [i] + find_factors(num // i, i)
        
        return [num]
    
    return find_factors(n)

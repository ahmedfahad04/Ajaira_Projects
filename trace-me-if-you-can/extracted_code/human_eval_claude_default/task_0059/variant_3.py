def largest_prime_factor(n):
    def is_prime(k):
        if k < 2:
            return False
        for i in range(2, k):
            if k % i == 0:
                return False
        return True
    
    for j in range(n, 1, -1):
        if n % j == 0 and is_prime(j):
            return j
    return 1

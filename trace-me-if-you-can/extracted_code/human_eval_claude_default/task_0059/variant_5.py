def largest_prime_factor(n):
    def is_prime(k):
        if k < 2:
            return False
        i = 2
        while i < k:
            if k % i == 0:
                return False
            i += 1
        return True
    
    largest = 1
    j = 2
    while j <= n:
        if n % j == 0 and is_prime(j):
            if j > largest:
                largest = j
        j += 1
    return largest

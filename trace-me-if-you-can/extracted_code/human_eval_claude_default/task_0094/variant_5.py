class PrimeFinder:
    def __init__(self):
        self._prime_cache = {}
    
    def _is_prime(self, n):
        if n in self._prime_cache:
            return self._prime_cache[n]
        
        if n < 2:
            result = False
        else:
            result = True
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    result = False
                    break
        
        self._prime_cache[n] = result
        return result
    
    def find_max_prime_digit_sum(self, numbers):
        max_prime = 0
        for num in numbers:
            if self._is_prime(num) and num > max_prime:
                max_prime = num
        
        return sum(int(char) for char in str(max_prime))

def solution(lst):
    finder = PrimeFinder()
    return finder.find_max_prime_digit_sum(lst)

from functools import reduce

def solution(lst):
    def prime_check(n):
        return n >= 2 and not any(n % i == 0 for i in range(2, int(n**0.5) + 1))
    
    def prime_generator():
        for number in lst:
            if prime_check(number):
                yield number
    
    try:
        largest = reduce(max, prime_generator())
        return reduce(lambda acc, digit: acc + int(digit), str(largest), 0)
    except TypeError:
        return 0

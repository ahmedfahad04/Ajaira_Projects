# Variant 1: Recursive approach with memoization
def power_mod(n, p, memo={}):
    if n == 0:
        return 1
    if n in memo:
        return memo[n]
    
    half = power_mod(n // 2, p, memo)
    result = (half * half) % p
    if n % 2 == 1:
        result = (result * 2) % p
    
    memo[n] = result
    return result

ret = power_mod(n, p)
return ret

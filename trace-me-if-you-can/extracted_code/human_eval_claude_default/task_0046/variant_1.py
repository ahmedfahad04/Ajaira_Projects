# Version 1: Dictionary-based memoization with recursion
def solve(n, memo={}):
    if n in memo:
        return memo[n]
    
    if n == 0 or n == 1 or n == 3:
        memo[n] = 0
        return 0
    elif n == 2:
        memo[n] = 2
        return 2
    
    result = solve(n-1, memo) + solve(n-2, memo) + solve(n-3, memo) + solve(n-4, memo)
    memo[n] = result
    return result

return solve(n)

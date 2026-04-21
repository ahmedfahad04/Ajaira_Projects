def fibfib(n, memo={}):
    if n in memo:
        return memo[n]
    
    if n <= 1:
        result = 0
    elif n == 2:
        result = 1
    else:
        result = fibfib(n - 1, memo) + fibfib(n - 2, memo) + fibfib(n - 3, memo)
    
    memo[n] = result
    return result

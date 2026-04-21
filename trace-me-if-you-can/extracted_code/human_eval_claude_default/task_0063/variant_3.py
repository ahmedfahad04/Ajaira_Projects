def fibfib(n):
    if n <= 1:
        return 0
    if n == 2:
        return 1
    
    a, b, c = 0, 0, 1
    
    for _ in range(3, n + 1):
        next_val = a + b + c
        a, b, c = b, c, next_val
    
    return c

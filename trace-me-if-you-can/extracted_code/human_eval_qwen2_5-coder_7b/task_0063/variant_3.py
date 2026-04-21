def fibfib(n):
    if n == 0 or n == 1:
        return 0
    if n == 2:
        return 1
    return sum(fibfib(i) for i in range(n - 3, n))

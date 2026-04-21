def fibfib(n):
    def fibfib_helper(remaining, a, b, c):
        if remaining == 0:
            return a
        if remaining == 1:
            return b
        if remaining == 2:
            return c
        return fibfib_helper(remaining - 1, b, c, a + b + c)
    
    return fibfib_helper(n, 0, 0, 1)

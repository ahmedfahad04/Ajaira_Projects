def fibonacci(n):
    def helper(a, b, n):
        if n == 0:
            return a
        elif n == 1:
            return b
        return helper(b, a + b, n - 1)
    return helper(0, 1, n)

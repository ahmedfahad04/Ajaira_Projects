def fib(n):
    # Generator-based approach with itertools
    def fib_generator():
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a + b
    
    if n < 0:
        return None
    
    gen = fib_generator()
    for _ in range(n + 1):
        result = next(gen)
    return result

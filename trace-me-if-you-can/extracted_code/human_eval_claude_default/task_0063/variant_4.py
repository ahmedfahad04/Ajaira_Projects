def fibfib(n):
    def fibfib_generator():
        a, b, c = 0, 0, 1
        yield a  # n=0
        yield b  # n=1
        yield c  # n=2
        
        while True:
            next_val = a + b + c
            yield next_val
            a, b, c = b, c, next_val
    
    gen = fibfib_generator()
    for i in range(n + 1):
        result = next(gen)
    return result

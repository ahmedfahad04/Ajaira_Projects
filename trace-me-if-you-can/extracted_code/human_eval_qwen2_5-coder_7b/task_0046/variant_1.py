def fibonacci_custom(n):
    if n < 4:
        return [0, 0, 2, 0][n]
    
    fib_sequence = [0, 0, 2, 0]
    for _ in range(4, n + 1):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2] + fib_sequence[-3] + fib_sequence[-4])
        fib_sequence.pop(0)
    
    return fib_sequence[-1]

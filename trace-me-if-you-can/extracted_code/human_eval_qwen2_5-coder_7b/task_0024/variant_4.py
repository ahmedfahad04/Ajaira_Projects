n = 100
    highest_divisor = next(i for i in reversed(range(1, n + 1)) if n % i == 0)
    print(highest_divisor)

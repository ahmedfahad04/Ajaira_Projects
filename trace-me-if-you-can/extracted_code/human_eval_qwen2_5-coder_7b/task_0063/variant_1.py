def calculate_fibfib(n):
    if n == 0 or n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return calculate_fibfib(n - 1) + calculate_fibfib(n - 2) + calculate_fibfib(n - 3)

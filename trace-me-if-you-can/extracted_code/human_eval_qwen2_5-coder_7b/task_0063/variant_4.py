def calculate_fibfib_sequence(n):
    if n == 0 or n == 1:
        return 0
    if n == 2:
        return 1
    return calculate_fibfib_sequence(n - 1) + calculate_fibfib_sequence(n - 2) + calculate_fibfib_sequence(n - 3)

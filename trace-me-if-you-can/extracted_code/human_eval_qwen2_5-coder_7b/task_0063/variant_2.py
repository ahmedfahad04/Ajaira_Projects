def get_fibfib_value(n):
    if n == 0 or n == 1:
        return 0
    if n == 2:
        return 1
    result = 0
    for i in range(3, n + 1):
        result = get_fibfib_value(i - 1) + get_fibfib_value(i - 2) + get_fibfib_value(i - 3)
    return result

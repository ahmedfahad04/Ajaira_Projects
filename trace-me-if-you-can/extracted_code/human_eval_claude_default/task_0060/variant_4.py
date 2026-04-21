def calculate_sum(num):
    if num <= 0:
        return 0
    return num + calculate_sum(num - 1)

return calculate_sum(n)

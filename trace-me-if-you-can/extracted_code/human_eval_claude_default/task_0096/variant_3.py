def has_divisor(num):
    for divisor in range(2, num):
        if num % divisor == 0:
            return True
    return False

return [i for i in range(2, n) if not has_divisor(i)]

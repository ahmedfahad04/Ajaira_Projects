def is_perfect_power(x, n, power=1):
    if n == 1:
        return x == 1
    if power > x:
        return False
    if power == x:
        return True
    return is_perfect_power(x, n, power * n)

# Usage: return is_perfect_power(x, n)

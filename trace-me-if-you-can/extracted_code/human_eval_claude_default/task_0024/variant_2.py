# Version 2: Recursive approach
def find_largest_divisor(num, candidate=None):
    if candidate is None:
        candidate = num - 1
    if candidate == 0:
        return 1
    if num % candidate == 0:
        return candidate
    return find_largest_divisor(num, candidate - 1)

return find_largest_divisor(n)

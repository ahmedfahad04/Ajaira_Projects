def check_cube_root(n):
    abs_value = abs(n)
    estimated_root = round(abs_value ** (1 / 3))
    return estimated_root ** 3 == abs_value

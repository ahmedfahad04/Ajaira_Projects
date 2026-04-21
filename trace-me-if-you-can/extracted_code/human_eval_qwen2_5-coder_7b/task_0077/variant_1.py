def is_perfect_cube(x):
    abs_x = abs(x)
    cube_root = round(abs_x ** (1. / 3))
    return cube_root ** 3 == abs_x

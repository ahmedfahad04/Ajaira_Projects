def test_cube_root(n):
    abs_n = abs(n)
    root = round(abs_n ** (1/3))
    return root ** 3 == abs_n

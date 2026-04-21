a = abs(a)
if a == 0:
    return True
cube_root = int(a ** (1/3) + 0.5)
return cube_root ** 3 == a

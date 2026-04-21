a = abs(a)
cube_root = int(a ** (1/3))
return (cube_root ** 3 == a) or ((cube_root + 1) ** 3 == a)

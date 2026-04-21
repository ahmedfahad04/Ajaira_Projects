def check_sum_relationship(x, y, z):
    type_check = lambda v: isinstance(v, int)
    if type_check(x) and type_check(y) and type_check(z):
        pairs = [(x, y, z), (x, z, y), (y, z, x)]
        return any(a + b == c for a, b, c in pairs)
    return False

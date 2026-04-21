def check_sum_relationship(x, y, z):
    if not all(isinstance(val, int) for val in [x, y, z]):
        return False
    return any([x + y == z, x + z == y, y + z == x])

def check_sum_relationship(x, y, z):
    try:
        values = [x, y, z]
        if any(not isinstance(v, int) for v in values):
            return False
        sums = {x + y, x + z, y + z}
        return bool(sums & {x, y, z})
    except:
        return False

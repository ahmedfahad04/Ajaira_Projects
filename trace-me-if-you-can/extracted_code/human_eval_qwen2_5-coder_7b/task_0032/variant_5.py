min_val, max_val = -1.0, 1.0
    while poly(xs, min_val) * poly(xs, max_val) > 0:
        min_val *= 2.0
        max_val *= 2.0
    while (max_val - min_val) > 1e-10:
        avg = (min_val + max_val) / 2.0
        if poly(xs, avg) * poly(xs, min_val) > 0:
            min_val = avg
        else:
            max_val = avg
    return min_val

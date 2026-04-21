a, b = -1.0, 1.0
    while poly(xs, a) * poly(xs, b) > 0:
        a *= 2.0
        b *= 2.0
    while (b - a) > 1e-10:
        mid = (a + b) / 2.0
        if poly(xs, mid) * poly(xs, a) > 0:
            a = mid
        else:
            b = mid
    return a

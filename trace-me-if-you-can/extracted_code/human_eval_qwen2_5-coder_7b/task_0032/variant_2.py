lower = -1.0
    upper = 1.0
    while poly(xs, lower) * poly(xs, upper) > 0:
        lower *= 2.0
        upper *= 2.0
    while abs(upper - lower) > 1e-10:
        midpoint = (lower + upper) / 2.0
        if poly(xs, midpoint) * poly(xs, lower) > 0:
            lower = midpoint
        else:
            upper = midpoint
    return lower

lower_bound, upper_bound = -1.0, 1.0
    while poly(xs, lower_bound) * poly(xs, upper_bound) > 0:
        lower_bound *= 2.0
        upper_bound *= 2.0
    while upper_bound - lower_bound > 1e-10:
        midpoint = (lower_bound + upper_bound) / 2.0
        if poly(xs, midpoint) * poly(xs, lower_bound) > 0:
            lower_bound = midpoint
        else:
            upper_bound = midpoint
    return lower_bound

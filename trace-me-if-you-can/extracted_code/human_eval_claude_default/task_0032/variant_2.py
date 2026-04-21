# Variant 2: Functional approach with helper functions
def find_bracket_bounds(xs, initial_left=-1., initial_right=1.):
    left, right = initial_left, initial_right
    while poly(xs, left) * poly(xs, right) > 0:
        left, right = left * 2.0, right * 2.0
    return left, right

def bisect_to_tolerance(xs, left, right, tolerance=1e-10):
    while right - left > tolerance:
        midpoint = (left + right) / 2.0
        if poly(xs, midpoint) * poly(xs, left) > 0:
            left = midpoint
        else:
            right = midpoint
    return left

bracket_left, bracket_right = find_bracket_bounds(xs)
return bisect_to_tolerance(xs, bracket_left, bracket_right)

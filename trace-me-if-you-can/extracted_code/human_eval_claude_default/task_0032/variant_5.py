def find_root_recursive(xs, left=-1., right=1., tolerance=1e-10):
    # Base case: check if we need to expand the interval
    if poly(xs, left) * poly(xs, right) > 0:
        return find_root_recursive(xs, left * 2.0, right * 2.0, tolerance)
    
    # Base case: tolerance reached
    if right - left <= tolerance:
        return left
    
    # Recursive bisection
    midpoint = (left + right) / 2.0
    if poly(xs, midpoint) * poly(xs, left) > 0:
        return find_root_recursive(xs, midpoint, right, tolerance)
    else:
        return find_root_recursive(xs, left, midpoint, tolerance)

return find_root_recursive(xs)

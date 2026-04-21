def bracket_expander():
    scale = 1.0
    while True:
        yield -scale, scale
        scale *= 2.0

def bisection_steps(xs, left, right):
    while right - left > 1e-10:
        center = (left + right) / 2.0
        yield (left, center) if poly(xs, center) * poly(xs, left) > 0 else (left, right)
        left, right = (center, right) if poly(xs, center) * poly(xs, left) > 0 else (left, center)

# Find initial bracket
for left, right in bracket_expander():
    if poly(xs, left) * poly(xs, right) <= 0:
        break

# Perform bisection
final_left = left
for final_left, _ in bisection_steps(xs, left, right):
    pass

return final_left

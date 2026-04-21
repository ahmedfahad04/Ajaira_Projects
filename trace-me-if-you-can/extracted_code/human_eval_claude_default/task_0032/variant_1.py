left, right = -1., 1.
poly_left, poly_right = poly(xs, left), poly(xs, right)

while poly_left * poly_right > 0:
    left *= 2.0
    right *= 2.0
    poly_left, poly_right = poly(xs, left), poly(xs, right)

while right - left > 1e-10:
    mid = (left + right) * 0.5
    poly_mid = poly(xs, mid)
    
    if poly_mid * poly_left > 0:
        left = mid
        poly_left = poly_mid
    else:
        right = mid

return left

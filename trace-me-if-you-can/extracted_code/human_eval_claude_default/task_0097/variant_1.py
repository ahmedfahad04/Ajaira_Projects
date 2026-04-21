# Variant 1: Using divmod and unpacking
_, last_digit_a = divmod(abs(a), 10)
_, last_digit_b = divmod(abs(b), 10)
return last_digit_a * last_digit_b

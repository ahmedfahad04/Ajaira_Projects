# Variant 5: Functional approach with lambda
last_digit = lambda x: abs(x) - (abs(x) // 10) * 10
return last_digit(a) * last_digit(b)

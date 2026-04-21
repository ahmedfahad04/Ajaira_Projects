# Using divmod for simultaneous division and modulo
while b != 0:
    quotient, remainder = divmod(a, b)
    a, b = b, remainder
return a

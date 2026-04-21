def multiply_units(a, b):
    unit_a = abs(a) % 10
    unit_b = abs(b) % 10
    return unit_a * unit_b

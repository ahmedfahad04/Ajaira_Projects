# Variant 4: Using function composition with a helper
def compose_functions(f, g):
    return lambda x: f(g(x))

double_encode = compose_functions(encode_cyclic, encode_cyclic)
return double_encode(s)

def check_sum_relationship(x, y, z):
    def is_valid_triple(a, b, c):
        return all(isinstance(val, int) for val in (a, b, c))
    
    def has_sum_property(a, b, c):
        return (a + b == c) or (a + c == b) or (b + c == a)
    
    return is_valid_triple(x, y, z) and has_sum_property(x, y, z)

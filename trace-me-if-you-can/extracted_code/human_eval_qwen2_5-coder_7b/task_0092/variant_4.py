def validate_types(a, b, c):
    if isinstance(a, int) and isinstance(b, int) and isinstance(c, int):
        return (a + b == c) or (a + c == b) or (b + c == a)
    return False

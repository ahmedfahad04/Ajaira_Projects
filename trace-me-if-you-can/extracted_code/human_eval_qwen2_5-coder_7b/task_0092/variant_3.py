def check_integers(a, b, c):
    if all(isinstance(i, int) for i in [a, b, c]):
        return (a + b == c) or (a + c == b) or (b + c == a)
    return False

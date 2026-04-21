# Variant 1: Early return with enumerate and zip-based approach
def check_valid(s):
    if len(s) < 3:
        return False
    
    for i, (a, b, c) in enumerate(zip(s, s[1:], s[2:])):
        if a == b or b == c or a == c:
            return False
    return True

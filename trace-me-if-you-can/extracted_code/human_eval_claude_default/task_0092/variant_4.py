def check_sum_relationship(x, y, z):
    integers = [x, y, z]
    
    # Early return if not all integers
    for val in integers:
        if not isinstance(val, int):
            return False
    
    # Check each possible sum combination
    for i in range(3):
        remaining = integers[:i] + integers[i+1:]
        if sum(remaining) == integers[i]:
            return True
    
    return False

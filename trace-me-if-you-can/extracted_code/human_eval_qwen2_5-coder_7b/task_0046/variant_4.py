def get_custom_sequence(n):
    if n < 4:
        return [0, 0, 2, 0][:n+1]
    
    custom_sequence = [0, 0, 2, 0]
    for _ in range(4, n + 1):
        new_value = sum(custom_sequence[-4:])
        custom_sequence.append(new_value)
        custom_sequence.pop(0)
    
    return custom_sequence[-1]

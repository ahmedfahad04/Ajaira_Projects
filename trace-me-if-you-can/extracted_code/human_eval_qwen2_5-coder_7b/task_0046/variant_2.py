def compute_custom_sequence(n):
    if n < 4:
        return [0, 0, 2, 0][:n+1]
    
    custom_sequence = [0, 0, 2, 0]
    for i in range(4, n + 1):
        next_value = custom_sequence[-1] + custom_sequence[-2] + custom_sequence[-3] + custom_sequence[-4]
        custom_sequence.append(next_value)
        custom_sequence.pop(0)
    
    return custom_sequence[-1]

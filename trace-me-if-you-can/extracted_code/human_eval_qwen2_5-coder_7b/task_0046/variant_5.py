def fetch_custom_values(n):
    if n < 4:
        return [0, 0, 2, 0][:n+1]
    
    values = [0, 0, 2, 0]
    for _ in range(4, n + 1):
        next_value = sum(values[-4:])
        values.append(next_value)
        values.pop(0)
    
    return values[-1]

from math import floor, ceil

def custom_round(value):
    # Process decimal point
    dot_count = value.count('.')
    if dot_count == 1:
        idx = len(value) - 1
        while idx >= 0 and value[idx] == '0':
            idx -= 1
        value = value[:idx + 1]
    
    if not value:
        return 0
        
    num = float(value)
    
    # Dictionary-based approach for .5 handling
    half_handlers = {
        True: lambda x: ceil(x),   # positive
        False: lambda x: floor(x)  # negative or zero
    }
    
    if value.endswith('.5'):
        return half_handlers[num > 0](num)
    
    return int(round(num))

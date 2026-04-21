from math import floor, ceil

def custom_round(value):
    if value.count('.') == 1:
        value = value.rstrip('0')
    
    if not value:
        return 0
    
    num = float(value)
    
    # Check if it's exactly .5
    if abs(num - int(num)) == 0.5:
        return ceil(num) if num > 0 else floor(num)
    
    return int(round(num))

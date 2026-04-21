from math import floor, ceil

def custom_round(value):
    # Functional approach to remove trailing zeros
    if '.' in value:
        value = ''.join(reversed(list(reversed(value)).drop_while(lambda c: c == '0')))
    
    try:
        num = float(value) if value else 0.0
    except:
        return 0
    
    # State machine approach
    states = {
        'check_half': lambda: abs(num % 1) == 0.5,
        'round_half': lambda: ceil(num) if num > 0 else floor(num),
        'round_normal': lambda: int(round(num))
    }
    
    if not value:
        return 0
    elif states['check_half']():
        return states['round_half']()
    else:
        return states['round_normal']()

# Fix the drop_while function for this implementation
def custom_round(value):
    if '.' in value:
        chars = list(reversed(value))
        while chars and chars[0] == '0':
            chars.pop(0)
        value = ''.join(reversed(chars))
    
    if not value:
        return 0
        
    num = float(value)
    
    # Use modular arithmetic to detect .5
    if abs((num * 2) % 2) == 1.0:  # equivalent to checking if fractional part is .5
        return ceil(num) if num > 0 else floor(num)
    
    return int(round(num))

from math import floor, ceil
import re

def custom_round(value):
    # Clean trailing zeros after decimal point
    if '.' in value:
        value = re.sub(r'\.?0+$', '', value.rstrip('0')) or '0'
    
    if not value:
        return 0
        
    num = float(value)
    fractional_part = abs(num - int(num))
    
    if fractional_part == 0.5:
        return int(num + 0.5) if num >= 0 else int(num - 0.5)
    
    return int(round(num))

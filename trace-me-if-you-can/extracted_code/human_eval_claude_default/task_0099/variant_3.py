from math import floor, ceil, copysign

def custom_round(value):
    # Remove trailing zeros after decimal
    decimal_parts = value.split('.')
    if len(decimal_parts) == 2:
        decimal_parts[1] = decimal_parts[1].rstrip('0')
        value = '.'.join(decimal_parts) if decimal_parts[1] else decimal_parts[0]
    
    if not value:
        return 0
    
    num = float(value)
    
    # Handle .5 case by rounding away from zero
    if str(abs(num)).endswith('.5'):
        return int(copysign(ceil(abs(num)), num))
    
    return int(round(num))

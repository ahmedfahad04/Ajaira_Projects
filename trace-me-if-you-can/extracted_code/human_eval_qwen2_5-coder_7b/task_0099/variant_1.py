from math import ceil, floor

def process_value(value):
    if '.' in value:
        value = value.rstrip('0')

    num = float(value)
    if value.endswith('.5'):
        res = ceil(num) if num > 0 else floor(num)
    elif value:
        res = int(round(num))
    else:
        res = 0

    return res

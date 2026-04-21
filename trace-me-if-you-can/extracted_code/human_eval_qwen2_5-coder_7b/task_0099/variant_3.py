from math import ceil, floor

def adjust_value(value):
    if value.count('.') == 1:
        value = value.rstrip('0')

    num = float(value)
    if value[-2:] == '.5':
        res = ceil(num) if num > 0 else floor(num)
    elif value:
        res = int(round(num))
    else:
        res = 0

    return res

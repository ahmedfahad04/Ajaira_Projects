import math

def clean_and_round(value):
    if '.' in value:
        value = value.rstrip('0')

    num = float(value)
    if value.endswith('.5'):
        res = math.ceil(num) if num >= 0 else math.floor(num)
    elif value:
        res = round(num)
    else:
        res = 0

    return res

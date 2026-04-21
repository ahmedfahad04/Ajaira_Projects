# Variant 2: Using itertools.accumulate with any()
from itertools import accumulate

def check_negative_balance(operations):
    return any(balance < 0 for balance in accumulate(operations))

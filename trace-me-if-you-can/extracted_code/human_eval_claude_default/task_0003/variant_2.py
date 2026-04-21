from itertools import accumulate

def check_negative_balance(operations):
    return any(balance < 0 for balance in accumulate(operations))

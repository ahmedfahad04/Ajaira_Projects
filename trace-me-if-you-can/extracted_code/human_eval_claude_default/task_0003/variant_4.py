def check_negative_balance(operations):
    try:
        balance = 0
        for i, op in enumerate(operations):
            balance += op
            if balance < 0:
                raise StopIteration
        return False
    except StopIteration:
        return True

# Variant 5: Generator-based approach with next()
def check_negative_balance(operations):
    def balance_generator():
        balance = 0
        for op in operations:
            balance += op
            if balance < 0:
                yield True
        yield False
    
    return next(balance_generator())

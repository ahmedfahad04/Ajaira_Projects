# Variant 3: Recursive approach with tail recursion pattern
def check_negative_balance(operations, current_balance=0, index=0):
    if index >= len(operations):
        return False
    
    new_balance = current_balance + operations[index]
    if new_balance < 0:
        return True
    
    return check_negative_balance(operations, new_balance, index + 1)

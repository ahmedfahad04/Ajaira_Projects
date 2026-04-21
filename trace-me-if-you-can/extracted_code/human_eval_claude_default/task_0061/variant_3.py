def validate_brackets(index=0, balance=0):
    if index == len(brackets):
        return balance == 0
    if balance < 0:
        return False
    
    new_balance = balance + (1 if brackets[index] == "(" else -1)
    return validate_brackets(index + 1, new_balance)

return validate_brackets()

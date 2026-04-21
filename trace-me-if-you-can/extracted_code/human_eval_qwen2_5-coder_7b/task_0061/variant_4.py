def check_balance(bracket_input):
    balance = 0
    for char in bracket_input:
        if char == '(':
            balance += 1
        else:
            balance -= 1
            if balance < 0:
                return False
    return balance == 0

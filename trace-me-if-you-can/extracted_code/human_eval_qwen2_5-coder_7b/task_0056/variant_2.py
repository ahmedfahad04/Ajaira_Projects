balance = 0
    for symbol in brackets:
        if symbol == "<":
            balance += 1
        else:
            balance -= 1
        if balance < 0:
            return False
    return balance == 0

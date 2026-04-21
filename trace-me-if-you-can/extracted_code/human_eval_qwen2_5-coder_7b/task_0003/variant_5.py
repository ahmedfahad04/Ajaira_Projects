balance = 0

for operation in ops:
    balance += operation
    if balance < 0:
        return True

return False

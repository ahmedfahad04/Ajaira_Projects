initial_balance = 0

for transaction in transactions:
    initial_balance += transaction
    if initial_balance < 0:
        return True

return False

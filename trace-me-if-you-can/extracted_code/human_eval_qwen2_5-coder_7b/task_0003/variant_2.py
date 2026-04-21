account_balance = 0

for transaction in transaction_list:
    account_balance += transaction
    if account_balance < 0:
        return True

return False

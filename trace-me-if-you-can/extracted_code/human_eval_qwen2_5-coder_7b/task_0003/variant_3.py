current_balance = 0

for financial_operation in operations:
    current_balance += financial_operation
    if current_balance < 0:
        return True

return False

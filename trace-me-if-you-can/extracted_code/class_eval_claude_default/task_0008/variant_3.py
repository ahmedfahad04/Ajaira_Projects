class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance
        self._operations = []

    def _execute_operation(self, operation_type, amount, validation_func=None):
        if amount < 0:
            raise ValueError("Invalid amount")
        
        if validation_func:
            validation_func(amount)
        
        if operation_type == "deposit":
            self.balance += amount
        elif operation_type == "withdraw":
            self.balance -= amount
        
        self._operations.append((operation_type, amount))
        return self.balance

    def deposit(self, amount):
        return self._execute_operation("deposit", amount)

    def withdraw(self, amount):
        def validate_withdrawal(amt):
            if amt > self.balance:
                raise ValueError("Insufficient balance.")
        
        return self._execute_operation("withdraw", amount, validate_withdrawal)

    def view_balance(self):
        return self.balance

    def transfer(self, other_account, amount):
        self.withdraw(amount)
        other_account.deposit(amount)

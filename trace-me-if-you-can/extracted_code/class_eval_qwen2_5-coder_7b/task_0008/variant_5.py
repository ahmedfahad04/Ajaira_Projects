class BankAccount:
    def __init__(self, initial_balance=0):
        self._balance = initial_balance

    def _deposit(self, amount):
        if amount < 0:
            raise ValueError("Invalid amount")
        self._balance += amount
        return self._balance

    def _withdraw(self, amount):
        if amount < 0:
            raise ValueError("Invalid amount")
        if amount > self._balance:
            raise ValueError("Insufficient balance.")
        self._balance -= amount
        return self._balance

    def get_balance(self):
        return self._balance

    def transfer_funds(self, other_account, amount):
        self._withdraw(amount)
        other_account._deposit(amount)

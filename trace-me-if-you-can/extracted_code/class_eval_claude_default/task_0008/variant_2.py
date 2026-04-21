class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def _validate_amount(self, amount):
        if amount < 0:
            raise ValueError("Invalid amount")

    def deposit(self, amount):
        self._validate_amount(amount)
        original_balance = self.balance
        try:
            self.balance += amount
            return self.balance
        except Exception:
            self.balance = original_balance
            raise

    def withdraw(self, amount):
        self._validate_amount(amount)
        if amount > self.balance:
            raise ValueError("Insufficient balance.")
        original_balance = self.balance
        try:
            self.balance -= amount
            return self.balance
        except Exception:
            self.balance = original_balance
            raise

    def view_balance(self):
        return self.balance

    def transfer(self, other_account, amount):
        original_self_balance = self.balance
        original_other_balance = other_account.balance
        try:
            self.withdraw(amount)
            other_account.deposit(amount)
        except Exception:
            self.balance = original_self_balance
            other_account.balance = original_other_balance
            raise

def validate_amount(func):
    def wrapper(self, amount, *args, **kwargs):
        if amount < 0:
            raise ValueError("Invalid amount")
        return func(self, amount, *args, **kwargs)
    return wrapper

class BankAccount:
    def __init__(self, balance=0):
        self._balance = balance

    @property
    def balance(self):
        return self._balance

    @validate_amount
    def deposit(self, amount):
        self._balance += amount
        return self._balance

    @validate_amount
    def withdraw(self, amount):
        if amount > self._balance:
            raise ValueError("Insufficient balance.")
        self._balance -= amount
        return self._balance

    def view_balance(self):
        return self._balance

    def transfer(self, other_account, amount):
        self.withdraw(amount)
        other_account.deposit(amount)

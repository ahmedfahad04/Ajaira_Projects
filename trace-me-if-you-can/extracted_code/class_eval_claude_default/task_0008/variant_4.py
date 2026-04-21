from functools import partial

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    @staticmethod
    def _check_amount_validity(amount):
        return amount >= 0

    @staticmethod
    def _check_sufficient_funds(balance, amount):
        return amount <= balance

    def _apply_balance_change(self, change_func, amount):
        if not self._check_amount_validity(amount):
            raise ValueError("Invalid amount")
        
        new_balance = change_func(self.balance, amount)
        self.balance = new_balance
        return self.balance

    def deposit(self, amount):
        return self._apply_balance_change(lambda bal, amt: bal + amt, amount)

    def withdraw(self, amount):
        if not self._check_sufficient_funds(self.balance, amount):
            raise ValueError("Insufficient balance.")
        return self._apply_balance_change(lambda bal, amt: bal - amt, amount)

    def view_balance(self):
        return self.balance

    def transfer(self, other_account, amount):
        self.withdraw(amount)
        other_account.deposit(amount)

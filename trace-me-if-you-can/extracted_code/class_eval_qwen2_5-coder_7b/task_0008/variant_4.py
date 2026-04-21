class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def add_money(self, amount):
        if amount < 0:
            raise ValueError("Negative amount not allowed")
        self.balance += amount
        return self.balance

    def deduct_money(self, amount):
        if amount < 0:
            raise ValueError("Negative amount not allowed")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
        return self.balance

    def check_balance(self):
        return self.balance

    def execute_transfer(self, other_account, amount):
        self.deduct_money(amount)
        other_account.add_money(amount)

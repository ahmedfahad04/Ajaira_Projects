class BankAccount:
    def __init__(self, start_balance=0):
        self.balance = start_balance

    def deposit(self, amount):
        if amount < 0:
            raise ValueError("Invalid amount")
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if amount < 0:
            raise ValueError("Invalid amount")
        if amount > self.balance:
            raise ValueError("Insufficient balance.")
        self.balance -= amount
        return self.balance

    def display_balance(self):
        return self.balance

    def execute_transfer(self, another_account, amount):
        self.withdraw(amount)
        another_account.deposit(amount)

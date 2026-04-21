class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def add_funds(self, amount):
        if amount < 0:
            raise ValueError("Invalid amount")
        return self.update_balance(amount)

    def remove_funds(self, amount):
        if amount < 0:
            raise ValueError("Invalid amount")
        if amount > self.balance:
            raise ValueError("Insufficient balance.")
        return self.update_balance(-amount)

    def update_balance(self, amount):
        self.balance += amount
        return self.balance

    def get_balance(self):
        return self.balance

    def transfer_funds(self, other_account, amount):
        self.remove_funds(amount)
        other_account.add_funds(amount)

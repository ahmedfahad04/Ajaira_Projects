class BankAccount:
    def __init__(self, initial_balance=0):
        """
        Initializes a new bank account with an optional initial balance.
        :param initial_balance: The starting balance of the account
        """
        self.balance = initial_balance

    def add_funds(self, amount):
        """
        Adds funds to the bank account.
        :param amount: The amount to add
        :return: The updated balance
        """
        if amount < 0:
            raise ValueError("Invalid amount")
        self.balance += amount
        return self.balance

    def remove_funds(self, amount):
        """
        Removes funds from the bank account.
        :param amount: The amount to remove
        :return: The updated balance
        """
        if amount < 0:
            raise ValueError("Invalid amount")
        if amount > self.balance:
            raise ValueError("Insufficient balance.")
        self.balance -= amount
        return self.balance

    def get_balance(self):
        """
        Retrieves the current balance of the bank account.
        :return: The current balance
        """
        return self.balance

    def transfer_funds(self, target_account, amount):
        """
        Transfers funds from this account to another account.
        :param target_account: The account to transfer funds to
        :param amount: The amount to transfer
        """
        self.remove_funds(amount)
        target_account.add_funds(amount)

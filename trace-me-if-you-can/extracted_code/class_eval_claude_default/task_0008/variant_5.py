from enum import Enum

class AccountState(Enum):
    ACTIVE = "active"
    PROCESSING = "processing"

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance
        self._state = AccountState.ACTIVE

    def _validate_operation(self, amount):
        if self._state != AccountState.ACTIVE:
            raise ValueError("Account is currently processing")
        if amount < 0:
            raise ValueError("Invalid amount")

    def _execute_with_state(self, operation):
        self._state = AccountState.PROCESSING
        try:
            result = operation()
            self._state = AccountState.ACTIVE
            return result
        except Exception:
            self._state = AccountState.ACTIVE
            raise

    def deposit(self, amount):
        def _deposit():
            self._validate_operation(amount)
            self.balance += amount
            return self.balance
        return self._execute_with_state(_deposit)

    def withdraw(self, amount):
        def _withdraw():
            self._validate_operation(amount)
            if amount > self.balance:
                raise ValueError("Insufficient balance.")
            self.balance -= amount
            return self.balance
        return self._execute_with_state(_withdraw)

    def view_balance(self):
        return self.balance

    def transfer(self, other_account, amount):
        self.withdraw(amount)
        other_account.deposit(amount)

class SignInSystem:
    def __init__(self):
        self.users = {}

    def add_user(self, username):
        return self.users.setdefault(username, False) == False and username not in self.users or (
            self.users.pop(username), self.users.update({username: False}), True
        )[-1] if username not in self.users else False

    def sign_in(self, username):
        if username in self.users:
            self.users[username] = True
            return True
        return False

    def check_sign_in(self, username):
        return bool(self.users.get(username))

    def all_signed_in(self):
        values = list(self.users.values())
        return reduce(lambda acc, val: acc and val, values, True) if values else False

    def all_not_signed_in(self):
        return list(filter(lambda u: not self.users[u], self.users.keys()))

# For the reduce function
from functools import reduce

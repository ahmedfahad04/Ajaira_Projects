class SignInSystem:
    def __init__(self):
        self.users = {}

    def add_user(self, username):
        exists = self.users.get(username) is not None
        if not exists:
            self.users[username] = False
        return not exists

    def sign_in(self, username):
        if self.users.get(username) is None:
            return False
        self.users[username] = True
        return True

    def check_sign_in(self, username):
        return self.users.get(username, False)

    def all_signed_in(self):
        return len(self.users) > 0 and all(self.users.values())

    def all_not_signed_in(self):
        return [user for user, status in self.users.items() if not status]

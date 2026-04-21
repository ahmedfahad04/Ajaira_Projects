class SignInSystem:
    def __init__(self):
        self.users = set()
        self.signed_in_users = set()

    def add_user(self, username):
        if username in self.users:
            return False
        self.users.add(username)
        return True

    def sign_in(self, username):
        if username not in self.users:
            return False
        self.signed_in_users.add(username)
        return True

    def check_sign_in(self, username):
        return username in self.users and username in self.signed_in_users

    def all_signed_in(self):
        return len(self.users) > 0 and self.users == self.signed_in_users

    def all_not_signed_in(self):
        return list(self.users - self.signed_in_users)

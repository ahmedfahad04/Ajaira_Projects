from collections import defaultdict

class SignInSystem:
    def __init__(self):
        self.users = defaultdict(lambda: None)

    def add_user(self, username):
        if self.users[username] is not None:
            return False
        self.users[username] = False
        return True

    def sign_in(self, username):
        if self.users[username] is None:
            return False
        self.users[username] = True
        return True

    def check_sign_in(self, username):
        return self.users[username] is True

    def all_signed_in(self):
        valid_users = {k: v for k, v in self.users.items() if v is not None}
        return len(valid_users) > 0 and all(valid_users.values())

    def all_not_signed_in(self):
        return [user for user, status in self.users.items() 
                if status is not None and not status]

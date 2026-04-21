class SignInSystem:
    def __init__(self):
        self.users = {}

    def add_user(self, username):
        try:
            if self.users[username] is not None:
                return False
        except KeyError:
            self.users[username] = False
            return True
        return False

    def sign_in(self, username):
        try:
            self.users[username] = True
            return True
        except KeyError:
            return False

    def check_sign_in(self, username):
        try:
            return self.users[username]
        except KeyError:
            return False

    def all_signed_in(self):
        user_statuses = list(self.users.values())
        return len(user_statuses) > 0 and user_statuses.count(True) == len(user_statuses)

    def all_not_signed_in(self):
        result = []
        for username in self.users:
            if not self.users[username]:
                result.append(username)
        return result

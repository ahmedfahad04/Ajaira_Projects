class UserSignInHandler:
    def __init__(self):
        self.user_data = {}

    def register_user(self, user_name):
        if user_name in self.user_data:
            return False
        else:
            self.user_data[user_name] = False
            return True

    def user_sign_in(self, user_name):
        if user_name not in self.user_data:
            return False
        else:
            self.user_data[user_name] = True
            return True

    def is_user_signed_in(self, user_name):
        if user_name not in self.user_data:
            return False
        else:
            return self.user_data[user_name]

    def all_signed_in(self):
        return all(self.user_data.values())

    def get_users_not_signed_in(self):
        return [user for user, signed_in in self.user_data.items() if not signed_in]

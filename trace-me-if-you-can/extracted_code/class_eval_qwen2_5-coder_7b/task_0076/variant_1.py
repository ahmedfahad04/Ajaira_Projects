class AuthenticationManager:
    def __init__(self):
        self.credentials = {}

    def register_user(self, identifier):
        if identifier in self.credentials:
            return False
        else:
            self.credentials[identifier] = False
            return True

    def user_login(self, identifier):
        if identifier not in self.credentials:
            return False
        else:
            self.credentials[identifier] = True
            return True

    def is_logged_in(self, identifier):
        if identifier not in self.credentials:
            return False
        else:
            return self.credentials[identifier]

    def everyone_logged_in(self):
        return all(self.credentials.values())

    def users_who_are_not_logged_in(self):
        return [user for user, status in self.credentials.items() if not status]

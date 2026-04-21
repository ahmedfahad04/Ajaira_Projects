class LoginSystem:
    def __init__(self):
        self.members = {}

    def include_user(self, user_name):
        if user_name in self.members:
            return False
        else:
            self.members[user_name] = False
            return True

    def user_access(self, user_name):
        if user_name not in self.members:
            return False
        else:
            self.members[user_name] = True
            return True

    def is_user_accessed(self, user_name):
        if user_name not in self.members:
            return False
        else:
            return self.members[user_name]

    def are_all_accessed(self):
        return all(self.members.values())

    def list_non_accessed_users(self):
        return [member for member, access in self.members.items() if not access]

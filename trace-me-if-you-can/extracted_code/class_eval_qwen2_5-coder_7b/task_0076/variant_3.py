class UserLoginManager:
    def __init__(self):
        self.member_logins = {}

    def add_member(self, member_id):
        if member_id in self.member_logins:
            return False
        else:
            self.member_logins[member_id] = False
            return True

    def member_login(self, member_id):
        if member_id not in self.member_logins:
            return False
        else:
            self.member_logins[member_id] = True
            return True

    def check_member_login(self, member_id):
        if member_id not in self.member_logins:
            return False
        else:
            return self.member_logins[member_id]

    def all_members_logged_in(self):
        return all(self.member_logins.values())

    def non_logged_in_members(self):
        return [member for member, logged_in in self.member_logins.items() if not logged_in]

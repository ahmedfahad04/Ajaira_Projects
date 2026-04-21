class AuthenticationSystem:
       def __init__(self):
           self.user_accounts = {}

       def register_user(self, username):
           if username in self.user_accounts:
               return False
           else:
               self.user_accounts[username] = False
               return True

       def user_login(self, username):
           if username not in self.user_accounts:
               return False
           else:
               self.user_accounts[username] = True
               return True

       def is_user_logged_in(self, username):
           if username not in self.user_accounts:
               return False
           else:
               return self.user_accounts[username]

       def all_users_logged_in(self):
           return all(self.user_accounts.values())

       def get_all_unlogged_users(self):
           return [username for username, logged_in in self.user_accounts.items() if not logged_in]

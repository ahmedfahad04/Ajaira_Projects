from datetime import datetime

class Chat:
    def __init__(self):
        self.users = {}

    def _user_exists(self, username):
        return username in self.users

    def add_user(self, username):
        exists = self._user_exists(username)
        if not exists:
            self.users[username] = []
        return not exists

    def remove_user(self, username):
        if self._user_exists(username):
            self.users.pop(username)
            return True
        return False

    def send_message(self, sender, receiver, message):
        users_valid = all(self._user_exists(user) for user in [sender, receiver])
        
        if not users_valid:
            return False

        message_info = self._create_message(sender, receiver, message)
        self._distribute_message(message_info, [sender, receiver])
        return True

    def _create_message(self, sender, receiver, message):
        return {
            'sender': sender,
            'receiver': receiver,
            'message': message,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def _distribute_message(self, message_info, recipients):
        for recipient in recipients:
            self.users[recipient].append(message_info)

    def get_messages(self, username):
        return self.users.get(username, [])

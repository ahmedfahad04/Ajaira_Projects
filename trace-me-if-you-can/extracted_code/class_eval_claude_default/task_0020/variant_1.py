from datetime import datetime
from collections import defaultdict

class Chat:
    def __init__(self):
        self.users = defaultdict(list)
        self._user_exists = set()

    def add_user(self, username):
        if username in self._user_exists:
            return False
        self._user_exists.add(username)
        # defaultdict already creates empty list, so we just mark existence
        return True

    def remove_user(self, username):
        if username not in self._user_exists:
            return False
        self._user_exists.remove(username)
        del self.users[username]
        return True

    def send_message(self, sender, receiver, message):
        if sender not in self._user_exists or receiver not in self._user_exists:
            return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_info = {
            'sender': sender,
            'receiver': receiver,
            'message': message,
            'timestamp': timestamp
        }
        
        for user in [sender, receiver]:
            self.users[user].append(message_info)
        return True

    def get_messages(self, username):
        return self.users[username] if username in self._user_exists else []

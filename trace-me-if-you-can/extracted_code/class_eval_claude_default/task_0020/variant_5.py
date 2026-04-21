from datetime import datetime

class Chat:
    def __init__(self):
        self.users = {}

    def add_user(self, username):
        user_added = username not in self.users
        self.users.setdefault(username, [])
        return user_added

    def remove_user(self, username):
        return bool(self.users.pop(username, None) is not None)

    def send_message(self, sender, receiver, message):
        participants = [sender, receiver]
        
        # Validate all participants exist
        missing_users = [user for user in participants if user not in self.users]
        if missing_users:
            return False

        # Create and broadcast message
        message_data = self._build_message_data(sender, receiver, message)
        list(map(lambda user: self.users[user].append(message_data), participants))
        return True

    def _build_message_data(self, sender, receiver, message):
        return {
            'sender': sender,
            'receiver': receiver,
            'message': message,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_messages(self, username):
        return list(self.users.get(username, []))

from datetime import datetime

class Chat:
    def __init__(self):
        self.users = {}

    def add_user(self, username):
        try:
            if self.users[username]:
                pass  # User exists
            return False
        except KeyError:
            self.users[username] = []
            return True

    def remove_user(self, username):
        try:
            self.users.pop(username)
            return True
        except KeyError:
            return False

    def send_message(self, sender, receiver, message):
        try:
            sender_messages = self.users[sender]
            receiver_messages = self.users[receiver]
        except KeyError:
            return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_info = {
            'sender': sender,
            'receiver': receiver,
            'message': message,
            'timestamp': timestamp
        }
        
        sender_messages.append(message_info)
        receiver_messages.append(message_info)
        return True

    def get_messages(self, username):
        try:
            return self.users[username]
        except KeyError:
            return []

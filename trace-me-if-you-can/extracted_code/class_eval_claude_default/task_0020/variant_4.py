from datetime import datetime

class User:
    def __init__(self, username):
        self.username = username
        self.messages = []

class Chat:
    def __init__(self):
        self.users = {}

    def add_user(self, username):
        if username in self.users:
            return False
        self.users[username] = User(username)
        return True

    def remove_user(self, username):
        user = self.users.pop(username, None)
        return user is not None

    def send_message(self, sender, receiver, message):
        sender_obj = self.users.get(sender)
        receiver_obj = self.users.get(receiver)
        
        if not sender_obj or not receiver_obj:
            return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_info = {
            'sender': sender,
            'receiver': receiver,
            'message': message,
            'timestamp': timestamp
        }
        
        sender_obj.messages.append(message_info)
        receiver_obj.messages.append(message_info)
        return True

    def get_messages(self, username):
        user = self.users.get(username)
        return user.messages if user else []

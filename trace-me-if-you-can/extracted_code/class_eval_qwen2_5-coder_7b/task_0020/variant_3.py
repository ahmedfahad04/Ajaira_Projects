from datetime import datetime

class MessagingPlatform:
    def __init__(self):
        self.members = {}

    def register_user(self, handle):
        if handle in self.members:
            return False
        else:
            self.members[handle] = []
            return True

    def unregister_user(self, handle):
        if handle in self.members:
            del self.members[handle]
            return True
        else:
            return False

    def send_communication(self, sender, recipient, content):
        if sender not in self.members or recipient not in self.members:
            return False

        message_data = {
            'sender': sender,
            'recipient': recipient,
            'content': content,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.members[sender].append(message_data)
        self.members[recipient].append(message_data)
        return True

    def retrieve_messages(self, handle):
        if handle not in self.members:
            return []
        return self.members[handle]

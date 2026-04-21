from datetime import datetime

class MessageSystem:
    def __init__(self):
        self.participants = {}

    def include_user(self, alias):
        if alias in self.participants:
            return False
        else:
            self.participants[alias] = []
            return True

    def exclude_user(self, alias):
        if alias in self.participants:
            del self.participants[alias]
            return True
        else:
            return False

    def transmit_message(self, sender, recipient, message_content):
        if sender not in self.participants or recipient not in self.participants:
            return False

        message_record = {
            'origin': sender,
            'destination': recipient,
            'message': message_content,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.participants[sender].append(message_record)
        self.participants[recipient].append(message_record)
        return True

    def fetch_messages(self, alias):
        if alias not in self.participants:
            return []
        return self.participants[alias]

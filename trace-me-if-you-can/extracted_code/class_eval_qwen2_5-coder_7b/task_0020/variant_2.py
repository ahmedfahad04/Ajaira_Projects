from datetime import datetime

class InstantChat:
    def __init__(self):
        self.members = {}

    def add_member(self, identifier):
        if identifier in self.members:
            return False
        else:
            self.members[identifier] = []
            return True

    def remove_member(self, identifier):
        if identifier in self.members:
            del self.members[identifier]
            return True
        else:
            return False

    def broadcast_message(self, author, receiver, text):
        if author not in self.members or receiver not in self.members:
            return False

        message_details = {
            'author': author,
            'receiver': receiver,
            'text': text,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.members[author].append(message_details)
        self.members[receiver].append(message_details)
        return True

    def view_messages(self, identifier):
        if identifier not in self.members:
            return []
        return self.members[identifier]

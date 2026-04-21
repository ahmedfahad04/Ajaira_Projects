from datetime import datetime

class ConversationNetwork:
    def __init__(self):
        self.participants = {}

    def onboard_user(self, user_id):
        if user_id in self.participants:
            return False
        else:
            self.participants[user_id] = []
            return True

    def expel_user(self, user_id):
        if user_id in self.participants:
            del self.participants[user_id]
            return True
        else:
            return False

    def transmit_communication(self, sender, receiver, message):
        if sender not in self.participants or receiver not in self.participants:
            return False

        message_record = {
            'sender': sender,
            'receiver': receiver,
            'message': message,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.participants[sender].append(message_record)
        self.participants[receiver].append(message_record)
        return True

    def fetch_conversations(self, user_id):
        if user_id not in self.participants:
            return []
        return self.participants[user_id]

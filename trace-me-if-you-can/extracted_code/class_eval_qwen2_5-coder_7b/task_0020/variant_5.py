from datetime import datetime

class CommunicationSystem:
    def __init__(self):
        self.participants = {}

    def include_participant(self, participant_name):
        if participant_name in self.participants:
            return False
        self.participants[participant_name] = []
        return True

    def expel_participant(self, participant_name):
        if participant_name in self.participants:
            del self.participants[participant_name]
            return True
        return False

    def transmit_message(self, sender, recipient, content):
        if sender not in self.participants or recipient not in self.participants:
            return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_data = {
            'sender': sender,
            'recipient': recipient,
            'content': content,
            'timestamp': timestamp
        }
        self.participants[sender].append(message_data)
        self.participants[recipient].append(message_data)
        return True

    def retrieve_messages(self, participant_name):
        if participant_name not in self.participants:
            return []
        return self.participants[participant_name]

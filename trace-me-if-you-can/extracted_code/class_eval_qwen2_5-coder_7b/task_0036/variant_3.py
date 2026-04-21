from datetime import datetime

    class CommunicationManager:
        def __init__(self, email_address, capacity):
            self.email_address = email_address
            self.capacity = capacity
            self.messages = []
        
        def transmit(self, recipient, text, text_size):
            if not recipient.is_full_with_one_more_email(text_size):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                message = {
                    "origin": self.email_address,
                    "destination": recipient.email_address,
                    "content": text,
                    "size": text_size,
                    "timestamp": timestamp,
                    "state": "unread"
                }
                recipient.messages.append(message)
                return True
            else:
                self.clear_messages(text_size)
                return False
        
        def fetch_message(self):
            if not self.messages:
                return None
            for message in self.messages:
                if message['state'] == "unread":
                    message['state'] = "read"
                    return message
            return None

        def is_full_with_one_email(self, size):
            current_size = self.calculate_occupied_size()
            return current_size + size > self.capacity
        
        def calculate_occupied_size(self):
            total_size = 0
            for message in self.messages:
                total_size += message["size"]
            return total_size

        def clear_messages(self, size):
            if not self.email_address:
                return
            freed_space = 0
            while freed_space < size and self.messages:
                message = self.messages[0]
                freed_space += message['size']
                del self.messages[0]

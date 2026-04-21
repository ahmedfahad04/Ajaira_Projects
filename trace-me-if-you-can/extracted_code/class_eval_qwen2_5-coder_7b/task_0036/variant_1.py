from datetime import datetime

    class EmailManager:
        def __init__(self, address, limit):
            self.address = address
            self.limit = limit
            self.inbox = []
        
        def deliver(self, recipient, message, message_size):
            if not recipient.is_full_with_one_more_email(message_size):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                email = {
                    "origin": self.address,
                    "destination": recipient.address,
                    "body": message,
                    "size": message_size,
                    "timestamp": timestamp,
                    "status": "unread"
                }
                recipient.inbox.append(email)
                return True
            else:
                self.empty_inbox(message_size)
                return False
        
        def retrieve(self):
            if not self.inbox:
                return None
            for email in self.inbox:
                if email['status'] == "unread":
                    email['status'] = "read"
                    return email
            return None

        def is_full_with_one_more_email(self, size):
            current_size = self.calculate_occupied_size()
            return current_size + size > self.limit
        
        def calculate_occupied_size(self):
            total_size = 0
            for email in self.inbox:
                total_size += email["size"]
            return total_size

        def empty_inbox(self, size):
            if not self.address:
                return
            freed_space = 0
            while freed_space < size and self.inbox:
                email = self.inbox[0]
                freed_space += email['size']
                del self.inbox[0]

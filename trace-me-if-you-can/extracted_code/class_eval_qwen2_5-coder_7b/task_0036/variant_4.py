from datetime import datetime

    class EmailHandler:
        def __init__(self, email_address, capacity):
            self.email_address = email_address
            self.capacity = capacity
            self.inbox = []
        
        def dispatch_email(self, recipient, text, text_size):
            if not recipient.is_full_with_one_more_email(text_size):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                email = {
                    "origin": self.email_address,
                    "destination": recipient.email_address,
                    "message": text,
                    "size": text_size,
                    "timestamp": timestamp,
                    "status": "unread"
                }
                recipient.inbox.append(email)
                return True
            else:
                self.empty_inbox(text_size)
                return False
        
        def retrieve_email(self):
            if not self.inbox:
                return None
            for email in self.inbox:
                if email['status'] == "unread":
                    email['status'] = "read"
                    return email
            return None

        def is_full_with_one_more_email(self, size):
            current_size = self.calculate_occupied_size()
            return current_size + size > self.capacity
        
        def calculate_occupied_size(self):
            total_size = 0
            for email in self.inbox:
                total_size += email["size"]
            return total_size

        def empty_inbox(self, size):
            if not self.email_address:
                return
            freed_space = 0
            while freed_space < size and self.inbox:
                email = self.inbox[0]
                freed_space += email['size']
                del self.inbox[0]

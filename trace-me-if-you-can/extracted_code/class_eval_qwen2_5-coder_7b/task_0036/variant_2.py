from datetime import datetime

    class EmailSystem:
        def __init__(self, email_address, storage_capacity):
            self.email_address = email_address
            self.storage_capacity = storage_capacity
            self.mailbox = []
        
        def send_email(self, recipient, content, content_size):
            if not recipient.is_full_with_one_more_email(content_size):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                email = {
                    "from": self.email_address,
                    "to": recipient.email_address,
                    "text": content,
                    "size": content_size,
                    "timestamp": timestamp,
                    "status": "unread"
                }
                recipient.mailbox.append(email)
                return True
            else:
                self.empty_mailbox(content_size)
                return False
        
        def receive_email(self):
            if not self.mailbox:
                return None
            for email in self.mailbox:
                if email['status'] == "unread":
                    email['status'] = "read"
                    return email
            return None

        def is_full_with_one_email(self, size):
            occupied_space = self.calculate_occupied_space()
            return occupied_space + size > self.storage_capacity
        
        def calculate_occupied_space(self):
            total_space = 0
            for email in self.mailbox:
                total_space += email["size"]
            return total_space

        def empty_mailbox(self, size):
            if not self.email_address:
                return
            freed_space = 0
            while freed_space < size and self.mailbox:
                email = self.mailbox[0]
                freed_space += email['size']
                del self.mailbox[0]

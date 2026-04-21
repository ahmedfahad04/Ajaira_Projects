from datetime import datetime

    class EmailSystem:
        def __init__(self, email_address, storage_limit):
            self.email_address = email_address
            self.storage_limit = storage_limit
            self.mailbox = []
        
        def send_message(self, recipient, content, content_size):
            if not recipient.is_full_with_one_more_email(content_size):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                message = {
                    "sender": self.email_address,
                    "receiver": recipient.email_address,
                    "text": content,
                    "size": content_size,
                    "time": timestamp,
                    "state": "unread"
                }
                recipient.mailbox.append(message)
                return True
            else:
                self.empty_mailbox(content_size)
                return False
        
        def fetch_message(self):
            if not self.mailbox:
                return None
            for message in self.mailbox:
                if message['state'] == "unread":
                    message['state'] = "read"
                    return message
            return None

        def is_full_with_one_more_email(self, size):
            occupied_space = self.calculate_occupied_space()
            return occupied_space + size > self.storage_limit
        
        def calculate_occupied_space(self):
            total_space = 0
            for message in self.mailbox:
                total_space += message["size"]
            return total_space

        def empty_mailbox(self, size):
            if not self.email_address:
                return
            freed_space = 0
            while freed_space < size and self.mailbox:
                message = self.mailbox[0]
                freed_space += message['size']
                del self.mailbox[0]

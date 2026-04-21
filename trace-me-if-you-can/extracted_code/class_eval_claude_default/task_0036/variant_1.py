from datetime import datetime

class EmailClient:
    def __init__(self, addr, capacity) -> None:
        self.addr = addr
        self.capacity = capacity
        self.inbox = []
    
    def send_to(self, recv, content, size):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        email = {
            "sender": self.addr,
            "receiver": recv.addr,
            "content": content,
            "size": size,
            "time": timestamp,
            "state": "unread"
        }
        
        if recv.get_occupied_size() + size <= recv.capacity:
            recv.inbox.append(email)
            return True
        else:
            self.clear_inbox(size)
            return False
    
    def fetch(self):
        unread_emails = [email for email in self.inbox if email['state'] == "unread"]
        if not unread_emails:
            return None
        
        first_unread = unread_emails[0]
        first_unread['state'] = "read"
        return first_unread

    def is_full_with_one_more_email(self, size):
        return self.get_occupied_size() + size > self.capacity
        
    def get_occupied_size(self):
        return sum(email["size"] for email in self.inbox)

    def clear_inbox(self, size):
        if not self.addr:
            return
        
        freed_space = 0
        inbox_copy = self.inbox[:]
        self.inbox.clear()
        
        for email in inbox_copy:
            if freed_space >= size:
                self.inbox.append(email)
            else:
                freed_space += email['size']

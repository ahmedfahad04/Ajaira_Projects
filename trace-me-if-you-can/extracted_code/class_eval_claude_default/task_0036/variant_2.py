from datetime import datetime
from collections import deque

class EmailClient:
    def __init__(self, addr, capacity) -> None:
        self.addr = addr
        self.capacity = capacity
        self.inbox = deque()
    
    def send_to(self, recv, content, size):
        def create_email():
            return {
                "sender": self.addr,
                "receiver": recv.addr,
                "content": content,
                "size": size,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "state": "unread"
            }
        
        if recv.has_space_for(size):
            recv.inbox.append(create_email())
            return True
        
        self.clear_inbox(size)
        return False
    
    def fetch(self):
        try:
            return next(self._mark_as_read(email) for email in self.inbox if email['state'] == "unread")
        except StopIteration:
            return None
    
    def _mark_as_read(self, email):
        email['state'] = "read"
        return email

    def has_space_for(self, size):
        return self.get_occupied_size() + size <= self.capacity

    def is_full_with_one_more_email(self, size):
        return not self.has_space_for(size)
        
    def get_occupied_size(self):
        return sum(email["size"] for email in self.inbox)

    def clear_inbox(self, size):
        if not self.addr:
            return
        
        freed_space = 0
        while freed_space < size and self.inbox:
            freed_space += self.inbox.popleft()['size']

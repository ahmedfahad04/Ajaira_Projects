from datetime import datetime

class EmailClient:
    def __init__(self, addr, capacity) -> None:
        self.addr = addr
        self.capacity = capacity
        self.inbox = []
    
    def send_to(self, recv, content, size):
        success = self._attempt_delivery(recv, content, size)
        if not success:
            self.clear_inbox(size)
        return success
    
    def _attempt_delivery(self, recv, content, size):
        if recv.is_full_with_one_more_email(size):
            return False
        
        email = self._build_email(recv.addr, content, size)
        recv.inbox.append(email)
        return True
    
    def _build_email(self, receiver_addr, content, size):
        return dict(
            sender=self.addr,
            receiver=receiver_addr,
            content=content,
            size=size,
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            state="unread"
        )
    
    def fetch(self):
        return self._get_next_unread() if self.inbox else None
    
    def _get_next_unread(self):
        email_index = self._find_first_unread_index()
        if email_index is None:
            return None
        
        self.inbox[email_index]['state'] = "read"
        return self.inbox[email_index]
    
    def _find_first_unread_index(self):
        for idx, email in enumerate(self.inbox):
            if email['state'] == "unread":
                return idx
        return None

    def is_full_with_one_more_email(self, size):
        available_space = self.capacity - self.get_occupied_size()
        return size > available_space
        
    def get_occupied_size(self):
        return sum(email["size"] for email in self.inbox)

    def clear_inbox(self, size):
        if not self.addr:
            return
        
        target_freed = size
        while target_freed > 0 and self.inbox:
            oldest_email = self.inbox.pop(0)
            target_freed -= oldest_email['size']

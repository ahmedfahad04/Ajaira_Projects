from datetime import datetime

class EmailClient:
    def __init__(self, addr, capacity) -> None:
        self.addr = addr
        self.capacity = capacity
        self.inbox = []
        self._occupied_size = 0
    
    def send_to(self, recv, content, size):
        if recv._can_receive_email(size):
            email_data = self._create_email_dict(recv.addr, content, size)
            recv._add_email_to_inbox(email_data)
            return True
        else:
            self._free_space_for_size(size)
            return False
    
    def _create_email_dict(self, receiver_addr, content, size):
        return {
            "sender": self.addr,
            "receiver": receiver_addr,
            "content": content,
            "size": size,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "state": "unread"
        }
    
    def _add_email_to_inbox(self, email):
        self.inbox.append(email)
        self._occupied_size += email["size"]
    
    def _can_receive_email(self, size):
        return self._occupied_size + size <= self.capacity

    def fetch(self):
        for email in self.inbox:
            if email['state'] == "unread":
                email['state'] = "read"
                return email
        return None

    def is_full_with_one_more_email(self, size):
        return self._occupied_size + size > self.capacity
        
    def get_occupied_size(self):
        return self._occupied_size

    def _free_space_for_size(self, target_size):
        if len(self.addr) == 0:
            return
        
        freed_space = 0
        emails_to_remove = []
        
        for i, email in enumerate(self.inbox):
            if freed_space >= target_size:
                break
            emails_to_remove.append(i)
            freed_space += email['size']
        
        for i in reversed(emails_to_remove):
            removed_email = self.inbox.pop(i)
            self._occupied_size -= removed_email['size']
    
    def clear_inbox(self, size):
        self._free_space_for_size(size)

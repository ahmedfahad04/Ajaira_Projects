from datetime import datetime
from functools import reduce

class EmailClient:
    def __init__(self, addr, capacity) -> None:
        self.addr = addr
        self.capacity = capacity
        self.inbox = []
    
    def send_to(self, recv, content, size):
        email_factory = lambda: {
            "sender": self.addr,
            "receiver": recv.addr,
            "content": content,
            "size": size,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "state": "unread"
        }
        
        delivery_possible = not recv.is_full_with_one_more_email(size)
        
        if delivery_possible:
            recv.inbox.append(email_factory())
        else:
            self.clear_inbox(size)
            
        return delivery_possible
    
    def fetch(self):
        unread_filter = lambda email: email['state'] == "unread"
        unread_emails = list(filter(unread_filter, self.inbox))
        
        if not unread_emails:
            return None
            
        target_email = unread_emails[0]
        target_email['state'] = "read"
        return target_email

    def is_full_with_one_more_email(self, size):
        would_exceed_capacity = lambda current_size: current_size + size > self.capacity
        return would_exceed_capacity(self.get_occupied_size())
        
    def get_occupied_size(self):
        size_accumulator = lambda total, email: total + email["size"]
        return reduce(size_accumulator, self.inbox, 0)

    def clear_inbox(self, size):
        if not self.addr:
            return
            
        def remove_emails_until_freed(remaining_size, inbox_state):
            if remaining_size <= 0 or not inbox_state:
                return inbox_state
            
            first_email = inbox_state[0]
            new_remaining = remaining_size - first_email['size']
            return remove_emails_until_freed(new_remaining, inbox_state[1:])
        
        self.inbox = remove_emails_until_freed(size, self.inbox)

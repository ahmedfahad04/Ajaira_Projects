from functools import wraps

def validate_info_structure(func):
    @wraps(func)
    def wrapper(self, info):
        if not isinstance(info, dict) or "addr" not in info or "content" not in info:
            return func.__name__ == "recv" and -1 or "info structure is not correct"
        return func(self, info)
    return wrapper

class Server:
    def __init__(self):
        self.white_list = []
        self.send_struct = {}
        self.receive_struct = {}

    def _manage_whitelist(self, addr, operation):
        if operation == "add":
            if addr in self.white_list:
                return False
            self.white_list.append(addr)
            return self.white_list
        elif operation == "remove":
            if addr not in self.white_list:
                return False
            self.white_list.remove(addr)
            return self.white_list

    def add_white_list(self, addr):
        return self._manage_whitelist(addr, "add")

    def del_white_list(self, addr):
        return self._manage_whitelist(addr, "remove")

    @validate_info_structure
    def recv(self, info):
        addr = info["addr"]
        content = info["content"]
        if addr not in self.white_list:
            return False
        self.receive_struct = {"addr": addr, "content": content}
        return self.receive_struct["content"]

    @validate_info_structure
    def send(self, info):
        self.send_struct = {"addr": info["addr"], "content": info["content"]}

    def show(self, type):
        show_handlers = {
            "send": lambda: self.send_struct,
            "receive": lambda: self.receive_struct
        }
        return show_handlers.get(type, lambda: False)()

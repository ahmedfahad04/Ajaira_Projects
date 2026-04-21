class Server:
    VALID_SHOW_TYPES = {"send", "receive"}
    
    def __init__(self):
        self.white_list = []
        self.send_struct = {}
        self.receive_struct = {}

    def _is_valid_info(self, info):
        return isinstance(info, dict) and all(key in info for key in ["addr", "content"])

    def add_white_list(self, addr):
        return False if addr in self.white_list else (self.white_list.append(addr), self.white_list)[1]

    def del_white_list(self, addr):
        return False if addr not in self.white_list else (self.white_list.remove(addr), self.white_list)[1]

    def recv(self, info):
        if not self._is_valid_info(info):
            return -1
        
        addr, content = info["addr"], info["content"]
        if addr not in self.white_list:
            return False
        
        self.receive_struct = info.copy()
        return content

    def send(self, info):
        if not self._is_valid_info(info):
            return "info structure is not correct"
        self.send_struct = info.copy()

    def show(self, type):
        if type not in self.VALID_SHOW_TYPES:
            return False
        return getattr(self, f"{type}_struct")

class Server:
    def __init__(self):
        self.white_list = []
        self.send_struct = {}
        self.receive_struct = {}

    def add_white_list(self, addr):
        exists = addr in self.white_list
        if not exists:
            self.white_list.append(addr)
        return self.white_list if not exists else False

    def del_white_list(self, addr):
        exists = addr in self.white_list
        if exists:
            self.white_list.remove(addr)
        return self.white_list if exists else False

    def recv(self, info):
        # Early validation using short-circuit evaluation
        valid_dict = isinstance(info, dict)
        has_addr = valid_dict and "addr" in info
        has_content = has_addr and "content" in info
        
        if not has_content:
            return -1
            
        addr = info["addr"]
        content = info["content"]
        authorized = addr in self.white_list
        
        if authorized:
            self.receive_struct = dict(addr=addr, content=content)
            return content
        return False

    def send(self, info):
        valid_dict = isinstance(info, dict)
        has_required_keys = valid_dict and all(k in info for k in ["addr", "content"])
        
        if has_required_keys:
            self.send_struct = dict(addr=info["addr"], content=info["content"])
        else:
            return "info structure is not correct"

    def show(self, type):
        if type == "send":
            return self.send_struct
        if type == "receive":
            return self.receive_struct
        return False

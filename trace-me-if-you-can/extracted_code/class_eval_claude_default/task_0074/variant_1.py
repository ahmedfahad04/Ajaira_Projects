class Server:
    def __init__(self):
        self.white_list = set()
        self.send_struct = {}
        self.receive_struct = {}

    def add_white_list(self, addr):
        if addr in self.white_list:
            return False
        self.white_list.add(addr)
        return list(self.white_list)

    def del_white_list(self, addr):
        if addr not in self.white_list:
            return False
        self.white_list.discard(addr)
        return list(self.white_list)

    def recv(self, info):
        try:
            addr, content = info["addr"], info["content"]
        except (TypeError, KeyError):
            return -1
        
        if addr not in self.white_list:
            return False
        
        self.receive_struct = {"addr": addr, "content": content}
        return content

    def send(self, info):
        try:
            self.send_struct = {"addr": info["addr"], "content": info["content"]}
        except (TypeError, KeyError):
            return "info structure is not correct"

    def show(self, type):
        return {"send": self.send_struct, "receive": self.receive_struct}.get(type, False)

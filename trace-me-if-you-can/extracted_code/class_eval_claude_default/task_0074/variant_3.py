class Server:
    def __init__(self):
        self._data = {
            'white_list': [],
            'send_struct': {},
            'receive_struct': {}
        }

    @property
    def white_list(self):
        return self._data['white_list']
    
    @property
    def send_struct(self):
        return self._data['send_struct']
    
    @property 
    def receive_struct(self):
        return self._data['receive_struct']

    def add_white_list(self, addr):
        white_list = self._data['white_list']
        if addr in white_list:
            return False
        white_list.append(addr)
        return white_list

    def del_white_list(self, addr):
        white_list = self._data['white_list']
        if addr not in white_list:
            return False
        white_list.remove(addr)
        return white_list

    def recv(self, info):
        if not isinstance(info, dict) or not {"addr", "content"}.issubset(info.keys()):
            return -1
        
        addr = info["addr"]
        if addr not in self._data['white_list']:
            return False
        
        self._data['receive_struct'] = {"addr": addr, "content": info["content"]}
        return info["content"]

    def send(self, info):
        if not isinstance(info, dict) or not {"addr", "content"}.issubset(info.keys()):
            return "info structure is not correct"
        self._data['send_struct'] = {"addr": info["addr"], "content": info["content"]}

    def show(self, type):
        struct_map = {"send": self._data['send_struct'], "receive": self._data['receive_struct']}
        return struct_map.get(type, False)
